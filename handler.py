import runpod
import os
import sys
import json
import traceback
import base64
import shutil  # 🔵 [新增] 用于删除坏掉的文件夹
from datetime import datetime
import torch
import torchaudio
import boto3
from botocore.client import Config
from supabase import create_client
from huggingface_hub import snapshot_download  # 🔵 [新增] 核心下载工具
import gc

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

# 资产路径（对应您仓库根目录的位置）
# 默认 Docker 路径为 /app
ASSETS_DIR = os.environ.get("ASSETS_DIR", "/app/assets")
PROMPT_TEXTS_FILE = os.environ.get("PROMPT_TEXTS_FILE", "/app/prompt_texts.json")

# 模型路径
# 关键：必须指向 Network Volume 挂载点
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
    惰性获取 TTS 模型 (增强版：使用 snapshot_download)
    """
    global _tts_model
    if _tts_model is None:
        print(f"🚀 Checking model integrity in: {MODEL_DIR}")
        
        # 1. 检查关键文件是否存在 (作为下载成功的标志)
        # FireRedTTS2 的核心权重文件
        required_files = ["config_llm.json", "codec.pt"] 
        is_complete = os.path.exists(MODEL_DIR) and any(
            os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files
        )

        if not is_complete:
            print("   📥 Model missing or incomplete. Starting intelligent download...")
            
            # 2. 清理残余 (解决 git exit code 128 的关键)
            # 如果文件夹存在但文件不齐，说明上次下载断了。虽然 snapshot_download 支持断点，
            # 但为了保险，如果发现是个空壳文件夹，直接删掉重来。
            if os.path.exists(MODEL_DIR) and not os.listdir(MODEL_DIR):
                 print("   🧹 Removing empty directory to prevent conflicts...")
                 os.rmdir(MODEL_DIR)

            try:
                # 3. 使用官方 SDK 下载 (支持断点续传，不会报错文件夹已存在)
                snapshot_download(
                    repo_id="FireRedTeam/FireRedTTS2",
                    local_dir=MODEL_DIR,
                    resume_download=True,
                    max_workers=8
                )
                print("   ✅ Download complete.")
            except Exception as e:
                print(f"   ❌ Download failed: {e}")
                # 抛出异常让 RunPod 重启，不要继续尝试加载坏模型
                raise e
        else:
            print("   📂 Model integrity check passed. Using cache.")

        # 4. 加载模型
        print(f"🔥 Loading FireRedTTS2 from {MODEL_DIR}...")
        
        # 确保代码库在 path 中
        # 假设 Dockerfile 已经安装了依赖，但我们这里显式添加路径作为兜底
        if "/app/FireRedTTS2_Code" not in sys.path:
            sys.path.append("/app/FireRedTTS2_Code")
            
        from fireredtts2.fireredtts2 import FireRedTTS2
            
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
    
    s1_text = texts_data.get('S1')
    s2_text = texts_data.get('S2')
    
    # 🔴 [关键修复] 添加 [S1]/[S2] 标签前缀，修复 AssertionError
    # 你的 JSON 里是纯文本，模型要求必须带标签
    s1_text_tagged = f"[S1]{s1_text}"
    s2_text_tagged = f"[S2]{s2_text}"

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

    # 返回带标签的文本
    return (refined_paths, [s1_text_tagged, s2_text_tagged])

# ==================== 核心处理逻辑 (同步) ====================

def run_tts_process(episode_id: str):
    """
    同步执行 TTS 任务 (分批推理 + 显存保护版)
    """
    print(f"🔄 Processing Episode ID: {episode_id}")
    
    # 🟢 [新增] 强制显存清理 (任务开始前)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    supabase = get_supabase_client()
    tts_model = get_tts_model()
    r2_client = get_r2_client()

    # 1. 获取任务数据
    response = supabase.table('episodes').select('*').eq('id', episode_id).execute()
    if not response.data:
        raise ValueError(f"Episode {episode_id} not found")
    
    episode = response.data[0]
    language = episode.get('language')
    if not language: raise ValueError("Language field is missing")

    script_content = episode.get('script_content', {})
    
    # 2. 更新状态为 'tts_processing'
    print(f"   ⏳ Updating status to 'tts_processing'...")
    supabase.table('episodes').update({'status': 'tts_processing'}).eq('id', episode_id).execute()

    # 3. 准备文本 (保持原有清洗逻辑)
    raw_dialogue = script_content.get('dialogue', [])
    text_list = []
    
    for i, d in enumerate(raw_dialogue):
        role = d.get('role', 'Guest')
        content = d.get('text', '')
        if content:
            content = content.strip()
            content = content.replace('[', '【').replace(']', '】')
        if not content: continue
        tag = '[S1]' if role == 'Host' else '[S2]'
        text_list.append(f"{tag}{content}")

    if not text_list:
        raise ValueError("Script dialogue is empty after cleaning")
        
    print(f"   📝 Prepared {len(text_list)} lines. Preview: {text_list[:2]}...")

    # 4. 🟢 [重构] 分批推理 (Batch Inference)
    print(f"   🎙️ Generating audio for {episode_id}...")
    prompt_wavs, prompt_texts = get_cloning_refs(language)
    
    try:
        BATCH_SIZE = 10  # 每次处理 10 句
        audio_segments = []
        
        # 使用 inference_mode 进一步节省显存
        with torch.inference_mode():
            for i in range(0, len(text_list), BATCH_SIZE):
                batch_texts = text_list[i : i + BATCH_SIZE]
                print(f"      Processing Batch {i//BATCH_SIZE + 1}/{(len(text_list)+BATCH_SIZE-1)//BATCH_SIZE}...")
                
                # 推理当前批次
                wav_batch = tts_model.generate_dialogue(
                    text_list=batch_texts,
                    prompt_wav_list=prompt_wavs,
                    prompt_text_list=prompt_texts,
                    temperature=0.7,
                    topk=20
                )
                
                # 收集结果 (注意维度处理)
                if isinstance(wav_batch, list):
                    # 如果模型返回列表，拼接成 Tensor
                    wav_batch = torch.cat(wav_batch, dim=1) if len(wav_batch) > 0 else torch.tensor([])
                
                # 确保是 CPU Tensor，防止占用显存
                audio_segments.append(wav_batch.cpu())
                
                # 🟢 [关键] 每批次后清理显存
                del wav_batch
                torch.cuda.empty_cache()
        
        # 拼接所有批次
        if not audio_segments:
            raise ValueError("No audio generated")
            
        print("      Merging audio segments...")
        final_wav = torch.cat(audio_segments, dim=1)

    except AssertionError as ae:
        print(f"   🔴 Model Assertion Error! Input text format might be wrong.")
        print(f"   🔴 Debug Text List: {json.dumps(text_list, ensure_ascii=False)}")
        raise ae
    except Exception as e:
        torch.cuda.empty_cache()
        raise e

    # 5. 保存并上传 R2
    sample_rate = 24000
    tmp_path = f"/tmp/{episode_id}.wav"
    torchaudio.save(tmp_path, final_wav, sample_rate)
    duration_seconds = final_wav.shape[-1] // sample_rate

    r2_key = f"podcasts/{episode_id}.wav"
    print(f"   ☁️ Uploading to R2: {r2_key}")
    with open(tmp_path, 'rb') as f:
        r2_client.put_object(
            Bucket=os.environ.get("R2_BUCKET_NAME"), 
            Key=r2_key, 
            Body=f, 
            ContentType='audio/wav'
        )
    
    # ✅ 直接使用全局变量 R2_PUBLIC_URL (它已经在文件头部处理过默认值和rstrip了)
    audio_url = f"{R2_PUBLIC_URL}/{r2_key}"

    # 6. 完成回写
    print(f"   ✅ Done. Updating DB status to 'completed'...")
    supabase.table('episodes').update({
        'audio_url': audio_url,
        'duration': int(duration_seconds),
        'status': 'completed',
        'tts_updated_at': datetime.utcnow().isoformat(),
        'retry_count': 0
    }).eq('id', episode_id).execute()
    
    # 清理
    if os.path.exists(tmp_path): os.remove(tmp_path)
    # 🟢 [新增] 强制显存清理 (任务结束后)
    del final_wav
    del audio_segments
    gc.collect()
    torch.cuda.empty_cache()

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
        
        # 调用同步函数
        result = run_tts_process(episode_id)
        
        return {"status": "success", "output": result}

    except Exception as e:
        print(f"🔴 [RunPod] Error processing episode {episode_id}: {str(e)}")
        traceback.print_exc()
        
        # 失败回写
        if episode_id:
            try:
                supabase = get_supabase_client()
                resp = supabase.table('episodes').select('retry_count').eq('id', episode_id).execute()
                if resp.data:
                    current_retry = resp.data[0].get('retry_count', 0)
                    if current_retry < 3:
                        supabase.table('episodes').update({
                            'status': 'queued',
                            'retry_count': current_retry + 1
                        }).eq('id', episode_id).execute()
                        print(f"   🔄 Episode {episode_id} re-queued (retry #{current_retry + 1})")
                    else:
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
