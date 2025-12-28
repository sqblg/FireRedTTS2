from fastapi import FastAPI
from pydantic import BaseModel
import torch
from fireredtts2.fireredtts2 import FireRedTTS2
import sys

app = FastAPI()

class TTSRequest(BaseModel):
    text_list: list[str]
    language: str = "zh"
    temperature: float = 0.7

# 全局模型（容器启动时加载一次）
tts_model = FireRedTTS2(
    pretrained_dir="/app/pretrained_models/FireRedTTS2",
    gen_type="dialogue",
    device="cuda"
)

@app.post("/generate")
async def generate(request: TTSRequest):
    rec_wavs = tts_model.generate_dialogue(
        text_list=request.text_list,
        prompt_wav_list=None,
        prompt_text_list=None,
        temperature=request.temperature,
        topk=20
    )

    duration = rec_wavs.shape[-1] // 24000

    import torchaudio
    import uuid
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"
    torchaudio.save(tmp_path, rec_wavs.detach().cpu(), 24000)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "status": "success",
        "audio_base64": audio_b64,
        "duration": duration,
        "language": request.language
    }
