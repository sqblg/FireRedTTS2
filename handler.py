import os
import asyncio
import json
import traceback
import base64
from datetime import datetime
import torch
import torchaudio
import runpod
from fireredtts2.fireredtts2 import FireRedTTS2

# ==================== 环境变量 ====================
# 请在 RunPod 控制台的 Environment Variables 中设置以下变量
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "blockfm-audio")
R2_REGION = os.environ.get("R2_REGION", "auto")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://audio.blockfm.io")

# 资产路径（对应您仓库根目录的位置）
ASSETS_DIR = os.environ.get("ASSETS_DIR", "./assets")
PROMPT_TEXTS_FILE = os.environ.get("PROMPT_TEXTS_FILE", "./prompt_texts.json")

# 语言映射
LANG_ISO_TO_NAME = {
    'zh': 'Chinese', 'en': 'English', 'ja': 'Japanese',
    'ko': 'Korean', 'de': 'German', 'fr': 'French', 'ru': 'Russian'
}

# ==================== 全局状态 ====================
_supabase_client = None
_r2_client = None
_tts_model = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client

def get_r2_client():
    global _r2_client
    if _r2_client is None:
        import boto3
        from botocore.client import Config
        _r2_client = boto3.client(
            's3',
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name=R2_REGION,
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            config=Config(signature_version='s3v4')
        )
    return _r2_client

def get_tts_model():
    global _tts_model
    if _tts_model is None:
        # 网络卷挂载路径（RunPod Serverless 固定为 /runpod-volume）
        # 对应您创建的 firered-storge 卷
        MODEL_PATH = "/runpod-volume/FireRedTTS2"
        
        # 检查关键模型文件是否存在（以 codec.pt 为例）
        if not os.path.exists(os.path.join(MODEL_PATH, "codec.pt")):
            print(f"🚀 首次运行：正在下载模型至持久化网络卷 {MODEL_PATH}...")
            os.makedirs("/runpod-volume", exist_ok=True)
            # 执行下载逻辑。由于网络卷持久化，此操作仅需执行一次
            os.system(f"git lfs install && git clone https://huggingface.co/FireRedTeam/FireRedTTS2 {MODEL_PATH}")
            print("✅ 模型下载完成并已存入持久化存储。")
        else:
            print(f"🚀 发现持久化模型，正在从网络卷热加载...")

        _tts_model = FireRedTTS2(
            pretrained_dir=MODEL_PATH,
            gen_type="dialogue",
            device="cuda"
        )
        print("✅ 模型加载成功")
    return _tts_model

def get_cloning_refs(language_iso: str):
    lang_name = LANG_ISO_TO_NAME.get(language_iso)
    if not lang_name:
        raise ValueError(f"Unsupported language: {language_iso}")

    with open(PROMPT_TEXTS_FILE, 'r', encoding='utf-8') as f:
        all_prompt_texts = json.load(f)
    
    texts_data = all_prompt_texts.get(lang_name)
    s1_text, s2_text = texts_data.get('S1'), texts_data.get('S2')
    s1_path = os.path.join(ASSETS_DIR, language_iso, "S1.mp3")
    s2_path = os.path.join(ASSETS_DIR, language_iso, "S2.mp3")
    
    refined_paths = []
    for p in [s1_path, s2_path]:
        if os.path.exists(p):
            refined_paths.append(p)
        else:
            alt_p = p.replace(".mp3", ".flac")
            if os.path.exists(alt_p):
                refined_paths.append(alt_p)
            else:
                raise FileNotFoundError(f"Missing asset: {p}")
    return (refined_paths, [s1_text, s2_text])

# ==================== 核心处理逻辑 ====================

async def run_tts_process(episode_id: str):
    supabase = get_supabase()
    tts_model = get_tts_model()
    r2_client = get_r2_client()

    response = supabase.table('episodes').select('*').eq('id', episode_id).execute()
    if not response.data:
        raise ValueError(f"Episode {episode_id} not found")
    
    episode = response.data[0]
    language = episode.get('language')
    script_content = episode.get('script_content', {})
    
    supabase.table('episodes').update({'status': 'tts_processing'}).eq('id', episode_id).execute()

    dialogue = script_content.get('dialogue', [])
    text_list = [f"{'[S1]' if d.get('role') == 'Host' else '[S2]'}{d.get('text')}" for d in dialogue]

    prompt_wavs, prompt_texts = get_cloning_refs(language)
    rec_wavs = tts_model.generate_dialogue(
        text_list=text_list,
        prompt_wav_list=prompt_wavs,
        prompt_text_list=prompt_texts,
        temperature=0.7, topk=20
    )

    sample_rate = 24000
    tmp_path = f"/tmp/{episode_id}.wav"
    torchaudio.save(tmp_path, rec_wavs.detach().cpu(), sample_rate)
    duration_seconds = rec_wavs.shape[-1] // sample_rate

    r2_key = f"podcasts/{episode_id}.wav"
    with open(tmp_path, 'rb') as f:
        r2_client.put_object(
            Bucket=R2_BUCKET_NAME, Key=r2_key, Body=f,
            ContentType='audio/wav'
        )
    
    audio_url = f"{R2_PUBLIC_URL.rstrip('/')}/{r2_key}"

    supabase.table('episodes').update({
        'audio_url': audio_url,
        'duration': int(duration_seconds),
        'status': 'completed',
        'tts_updated_at': datetime.utcnow().isoformat()
    }).eq('id', episode_id).execute()

    return {"audio_url": audio_url}

# ==================== RunPod Handler ====================

def handler(job):
    try:
        job_input = job["input"]
        episode_id = job_input.get("episode_id")
        
        if not episode_id:
            return {"error": "Missing episode_id"}

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_tts_process(episode_id))
        
        return {"status": "success", "output": result}

    except Exception as e:
        print(f"🔴 Error: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
