from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import whisper
import tempfile
import os
import subprocess
import traceback
import random
from pathlib import Path
from fastapi import Request
import numpy as np
import time

app = FastAPI()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load model at startup (size configurable via WHISPER_MODEL env var)
model = None
MODEL_NAME = os.getenv('WHISPER_MODEL', 'small')

FAKE_RESPONSES = [
    "你好，我在。刚才我清楚地听见了你的声音。",
    "我正在思考你说的话，这对我来说很有意义。",
    "你可以慢慢说，我会一直在这里听着。",
    "这听起来很有趣，也许我们可以继续聊下去。",
]

@app.on_event('startup')
def load_model():
    global model
    print(f'Loading whisper model: {MODEL_NAME} ...')
    model = whisper.load_model(MODEL_NAME, download_root=str(PROJECT_ROOT / 'models'))
    print('Model loaded')

# 前端（使用基于 server 目录的静态目录）
static_dir = BASE_DIR / 'static'
if not static_dir.exists():
    print(f'Warning: static directory not found at {static_dir}')
app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')


@app.get('/')
def index():
    index_path = static_dir / 'index.html'
    if not index_path.exists():
        # fall back to project root index.html
        index_path = PROJECT_ROOT / 'index.html'
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f'index.html not found')
    return FileResponse(str(index_path))


@app.post('/api/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    print(f"Received file: {audio.filename}, content type: {audio.content_type}")

    tmp_input = None
    tmp_wav = None
    try:
        data = await audio.read()

        # write uploaded data to a temp file (binary)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1] or '.blob', mode='wb') as f:
            f.write(data)
            tmp_input = f.name

        print(f"Saved temporary file at: {tmp_input}")

        # convert to 16k mono wav using ffmpeg for more consistent transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as f2:
            tmp_wav = f2.name

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', tmp_input,
            '-ac', '1', '-ar', '16000', tmp_wav
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            print('ffmpeg not found:', e)
            raise HTTPException(status_code=500, detail='ffmpeg 未安装或不在 PATH 中；请安装 ffmpeg 并重试。')
        except subprocess.CalledProcessError as e:
            print('ffmpeg conversion failed:', e.stderr.decode(errors='ignore'))
            raise HTTPException(status_code=500, detail='音频转换失败（ffmpeg 错误），请检查上游音频格式。')

        # transcribe using whisper model
        try:
            result = model.transcribe(tmp_wav, language='zh')
        except Exception as e:
            print('transcribe error:', e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail='转录失败，查看服务器日志以获取更多信息')

        return {'text': result.get('text', '')}
    finally:
        # cleanup temp files
        for p in (tmp_input, tmp_wav):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    print(f'Removed temp file: {p}')
            except Exception as e:
                print('Failed to remove temp file', p, e)


@app.post("/api/chat/stream")
async def fake_chat_stream(request: Request):
    """
    Accepts either JSON {"text": "..."} or form-data with field 'text'.
    Returns a server-sent event stream (SSE) that yields the answer character-by-character.
    """
    text = None

    # Try to parse JSON first
    try:
        content_type = request.headers.get('content-type', '')
        if 'application/json' in content_type:
            body = await request.json()
            if isinstance(body, dict):
                text = body.get('text')
                print('chat stream body text:', text)
    except Exception as e:
        print('Failed to parse request body for /api/chat/stream:', e)

    if not text:
        # Return a clear validation error instead of FastAPI's generic 422
        raise HTTPException(status_code=422, detail="Request must include 'text' (JSON or form-data)")

    if isinstance(text, dict):
        answer = text.get('text')
        print('answer:', answer)

    async def generator():
        for ch in answer:
            yield f"data: {ch}\n\n"
            await asyncio.sleep(random.uniform(0.03, 0.12))

    return StreamingResponse(
        generator(),
        media_type="text/event-stream"
    )


SAMPLE_RATE = 16000
def fake_tts_stream(text: str):
    """
    模拟 TTS：
    - 不是真语音
    - 但是真·音频流（PCM 16bit）
    """
    # Produce more expressive synthetic audio with simple DSP:
    # - variable per-character duration
    # - pitch changes (glide + vibrato)
    # - amplitude envelope (attack/decay)
    # - second harmonic and subtle noise to make timbre richer
    # - occasional pauses (breaths)

    base_freq = 220.0
    rng = np.random.default_rng(abs(hash(text)) % (2**32))

    for i, ch in enumerate(text):
        # small chance of a short pause for punctuation or breath
        if ch in '.，,!?。！？' and rng.random() < 0.8:
            pause_dur = 0.12 + rng.random() * 0.18
            t = np.linspace(0, pause_dur, int(SAMPLE_RATE * pause_dur), endpoint=False)
            silence = np.zeros_like(t)
            pcm16 = (silence * 32767).astype(np.int16)
            yield pcm16.tobytes()
            time.sleep(pause_dur * 0.25)

        # duration and expressive modifiers
        duration_per_char = 0.06 + rng.random() * 0.12
        # pitch varies by position and random small offset
        freq = base_freq * (0.9 + 0.3 * rng.random()) * (1.0 + 0.02 * np.sin(i))

        # vibrato + glide
        vibrato_rate = 5.0 + rng.random() * 6.0
        vibrato_depth = 0.003 + rng.random() * 0.01
        glide = (0.98 + rng.random() * 0.04)

        t = np.linspace(0, duration_per_char, int(SAMPLE_RATE * duration_per_char), endpoint=False)

        # instantaneous frequency with vibrato and slow glide
        inst_freq = freq * (glide ** (t * 10)) * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
        phase = 2 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE
        carrier = np.sin(phase)

        # second harmonic and subtle noise for timbre
        harmonic = 0.4 * np.sin(2 * phase)
        noise = (rng.standard_normal(len(t)) * 0.012)

        # amplitude envelope (attack/decay)
        attack = int(0.01 * SAMPLE_RATE)
        release = int(0.03 * SAMPLE_RATE)
        env = np.ones_like(t)
        if len(t) > 0:
            if attack < len(t):
                env[:attack] = np.linspace(0.0, 1.0, attack)
            if release < len(t):
                env[-release:] = np.linspace(1.0, 0.0, release)

        # expressive amplitude based on position and randomness
        amp = 0.15 + 0.05 * rng.random() + 0.1 * np.clip(np.sin(i * 0.4), 0, 1)

        wave = amp * env * (carrier + harmonic) + noise * amp

        # occasionally add a stronger emphasis (like a stressed syllable)
        if rng.random() < 0.08:
            emphasize_len = int(0.02 * SAMPLE_RATE)
            if emphasize_len < len(wave):
                wave[:emphasize_len] += 0.08 * np.sin(2 * np.pi * (freq * 1.2) * t[:emphasize_len])

        pcm16 = np.clip(wave * 32767, -32768, 32767).astype(np.int16)

        # yield in moderately sized chunks to help smooth analyser updates
        chunk_size = int(0.05 * SAMPLE_RATE)  # 50ms chunks
        for start in range(0, len(pcm16), chunk_size):
            end = min(start + chunk_size, len(pcm16))
            yield pcm16[start:end].tobytes()
            # simulate asynchronous generation latency (small)
            time.sleep(0.01 + rng.random() * 0.02)


@app.get("/api/tts-stream")
def tts_stream(text: str):
    return StreamingResponse(
        fake_tts_stream(text),
        media_type="audio/pcm"
    )