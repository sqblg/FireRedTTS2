import os
import subprocess
import traceback
from datetime import datetime
import torch
import torchaudio
import runpod
from fireredtts2.fireredtts2 import FireRedTTS2

# ==================== 环境变量 ====================
# 请在 RunPod 控制台的 Environment Variables 中设置这些值
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "blockfm-audio")
R2_REGION = os.environ.get("R2_REGION", "auto")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "https://audio.blockfm.io")

# 资产路径 (对齐 Dockerfile 的 WORKDIR /app)
ASSETS_DIR = os.environ.get("ASSETS_DIR", "/app/assets")
PROMPT_TEXTS_FILE = os.environ.get("PROMPT_TEXTS_FILE", "/app/prompt_texts.json")

# 模型路径 (对齐 Dockerfile 的 ENV MODEL_DIR=/runpod-volume/FireRedTTS2)
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/FireRedTTS2")

LANG_ISO_TO_NAME = {
    'zh': 'Chinese', 'en': 'English', 'ja': 'Japanese',
    'ko': 'Korean', 'de': 'German', 'fr': 'French', 'ru': 'Russian'
}

# ==================== 全局状态 ====================
_supabase_client = None
_r2_client = None
_tts_model = None

def get_supabase_client():
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
    """
    🔵 [关键修正] 运行时模型下载
    如果 Network Volume 为空，自动从 HuggingFace Clone
    """
    global _tts_model
    if _tts_model is None:
        # 1. 检查 Volume 是否已有模型
        if not os.path.exists(MODEL_DIR):
            print(f"📥 Model not found in Volume at {MODEL_DIR}. Downloading...")
            os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
            
            try:
                # 使用 git-lfs 拉取大文件
                subprocess.run(["git", "lfs", "install"], check=True, env=os.environ.copy())
                subprocess.run([
                    "git", "clone", 
                    "https://huggingface.co/FireRedTeam/FireRedTTS2", 
                    MODEL_DIR
                ], check=True, env=os.environ.copy())
                print(f"   ✅ Model downloaded successfully to {MODEL_DIR}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to download model: {e}")
                raise
        
        # 2. 加载模型
        print(f"🚀 Loading FireRedTTS2 model from {MODEL_DIR}...")
        _tts_model = FireRedTTS2(
            pretrained_dir=MODEL_DIR,
            gen_type="dialogue",
            device="cuda"
        )
        print("✅ Model loaded successfully")
    return _tts_model

def get_cloning_refs(language_iso: str):
    lang_name = LANG_ISO_TO_NAME.get(language_iso)
    if not lang_name:
        raise ValueError(f"Unsupported language: {language_iso}")

    if not os.path.exists(PROMPT_TEXTS_FILE):
        raise FileNotFoundError(f"Missing prompt texts file: {PROMPT_TEXTS_FILE}")

    with open(PROMPT_TEXTS_FILE, 'r', encoding='utf-8') as f:
        all_prompt_texts = json.load(f)
    
    texts_data = all_prompt_texts.get(lang_name)
    if not texts_data:
        raise ValueError(f"Prompt texts not found for: {lang_name}")
    
    s1_text, s2_text = texts_data.get('S1'), texts_data.get('S2')
    s1_path = os.path.join(ASSETS_DIR, language_iso, "S1.mp3")
    s2_path = os.path.join(ASSETS_DIR, language_iso, "S2.mp3")
    
    # 检查文件是否存在并处理后缀
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

# ==================== 核心处理逻辑 (同步) ====================

def run_tts_process(episode_id: str):
    """
    同步执行 TTS 任务
    """
    supabase = get_supabase_client()
    tts_model = get_tts_model()
    r2_client = get_r2_client()

    # 1. 获取任务数据
    response = supabase.table('episodes').select('*').eq('id', episode_id).execute()
    if not response.data:
        raise ValueError(f"Episode {episode_id} not found")
    
    episode = response.data[0]
    language = episode.get('language')
    script_content = episode.get('script_content', {})
    
    # 2. 🔵 [关键] 更新状态为 'tts_processing' (咬合后端逻辑)
    print(f"   ⏳ Updating status to 'tts_processing'...")
    supabase.table('episodes').update({'status': 'tts_processing'}).eq('id', episode_id).execute()

    # 3. 准备文本
    dialogue = script_content.get('dialogue', [])
    text_list = [f"[{'S1' if d.get('role') == 'Host' else 'S2'}]{d.get('text')}" for d in dialogue]

    # 4. 推理
    print(f"   🎙️ Generating audio for {episode_id}...")
    prompt_wavs, prompt_texts = get_cloning_refs(language)
    rec_wavs = tts_model.generate_dialogue(
        text_list=text_list,
        prompt_wav_list=prompt_wavs,
        prompt_text_list=prompt_texts,
        temperature=0.7,
        topk=20
    )

    # 5. 保存并上传 R2
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

    # 6. 完成回写 (咬合后端 types.ts)
    print(f"   ✅ RAG Upload complete. Updating DB status to 'completed'...")
    supabase.table('episodes').update({
        'audio_url': audio_url,
        'duration': int(duration_seconds),
        'status': 'completed',
        'tts_updated_at': datetime.utcnow().isoformat()
    }).eq('id', episode_id).execute()

    return {"audio_url": audio_url}

# ==================== RunPod Handler ====================

def handler(job):
    """
    RunPod Serverless 入口函数
    """
    episode_id = None
    try:
        job_input = job["input"]
        episode_id = job_input.get("episode_id")
        
        if not episode_id:
            return {"error": "Missing episode_id"}

        print(f"\n🔥 [RunPod] Starting TTS job for episode {episode_id}")
        
        # 🔵 [修正] 调用同步函数
        result = run_tts_process(episode_id)
        
        return {"status": "success", "output": result}

    except Exception as e:
        print(f"🔴 [RunPod] Error processing episode {episode_id}: {str(e)}")
        traceback.print_exc()
        
        # 🔵 [关键修复] 显式回写数据库状态为 'failed'
        # 原因：防止数据永久卡在 'tts_processing'，允许后端 Cron 逻辑重试
        if episode_id:
            try:
                supabase = get_supabase_client()
                
                # 获取当前的 retry_count
                resp = supabase.table('episodes').select('retry_count').eq('id', episode_id).execute()
                if resp.data:
                    current_retry = resp.data[0].get('retry_count', 0)
                    
                    if current_retry < 3:
                        # 小于 3 次，重置为 queued 并增加计数
                        supabase.table('episodes').update({
                            'status': 'queued',
                            'retry_count': current_retry + 1
                        }).eq('id', episode_id).execute()
                        print(f"   🔄 Episode {episode_id} re-queued (retry #{current_retry + 1})")
                    else:
                        # 超过 3 次，标记为永久失败
                        supabase.table('episodes').update({
                            'status': 'failed'
                        }).eq('id', episode_id).execute()
                        print(f"   ❌ Episode {episode_id} marked as failed (max retries)")
                        
            except Exception as db_err:
                print(f"   ❌ Failed to update DB status: {db_err}")
        
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # 启动 RunPod Serverless Worker
    runpod.serverless.start({"handler": handler})
