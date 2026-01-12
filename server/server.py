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
import shutil
# Optional TTS / torch support: import lazily and tolerate absence so the server
# still starts when the environment doesn't have those packages installed.
tts_available = False
tts_engine = None
try:
    from TTS.api import TTS
    tts_available = True
    print('TTS package available')
except Exception as e:
    print('TTS package not available; continuing without it:', e)

# Torch safe-global allowlisting is helpful when loading some TTS checkpoints.
# Only attempt if torch is present.
try:
    import torch
    import collections
    try:
        # allowlist common globals used in checkpoints
        torch.serialization.add_safe_globals([collections.defaultdict, dict])
        print("Added collections.defaultdict to torch safe globals")
    except Exception as e:
        print("Warning: couldn't add collections.defaultdict to safe globals:", e)

    try:
        # 先导入定义 RAdam 的模块，再把类加入 safe globals（如果可用）
        from TTS.utils.radam import RAdam
        try:
            torch.serialization.add_safe_globals([RAdam])
            print("Added RAdam to torch safe globals")
        except Exception as e:
            print("Warning: couldn't add safe globals for RAdam:", e)
    except Exception:
        # not fatal
        pass
except Exception as e:
    print('torch not available or failed to import; skipping torch-safe-global setup:', e)

app = FastAPI()

# Detect ffmpeg availability so we can choose to stream raw PCM or return WAV
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None
if not FFMPEG_AVAILABLE:
    print('ffmpeg not found in PATH: server will return WAV container when possible')

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
    else:
        answer = text

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

    # Real TTS: synthesize to a temp WAV using local TTS engine, then stream 16k PCM via ffmpeg
    global tts_engine
    # use provided text when present, otherwise use a default sentence
    synth_text = text if text and str(text).strip() else "这是一个本地 TTS 测试语音。"

    # lazy init TTS engine if available; if not available, fall back to simple silence
    global tts_engine, tts_available
    if not tts_available:
        # no TTS package installed in this environment — yield short silence as fallback
        silence = (np.zeros(int(0.35 * SAMPLE_RATE))).astype(np.int16)
        yield silence.tobytes()
        return

    if tts_engine is None:
        try:
            tts_engine = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False)
        except Exception as e:
            print('Failed to initialize TTS engine:', e)
            # disable further attempts and fallback to silence
            tts_available = False
            silence = (np.zeros(int(0.35 * SAMPLE_RATE))).astype(np.int16)
            yield silence.tobytes()
            return

    tmp_wav = None
    proc = None
    try:
        # synthesize WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            tmp_wav = f.name

        tts_engine.tts_to_file(text=synth_text, file_path=tmp_wav)
        print(f'Synthesized TTS to temporary WAV: {tmp_wav}')

        # convert to 16k PCM signed 16 little-endian via ffmpeg to stdout
        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', tmp_wav,
            '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(SAMPLE_RATE), '-'
        ]

        # If ffmpeg isn't available on this host, return the WAV container as a fallback
        # (the client will detect content-type and play accordingly).
        if not FFMPEG_AVAILABLE:
            with open(tmp_wav, 'rb') as wf:
                while True:
                    chunk = wf.read(4096)
                    if not chunk:
                        break
                    yield chunk
            return

        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            # race condition: ffmpeg disappeared between check and run; fall back to WAV
            with open(tmp_wav, 'rb') as wf:
                while True:
                    chunk = wf.read(4096)
                    if not chunk:
                        break
                    yield chunk
            return

        # stream from process stdout in chunks
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk

        # wait for process to finish
        if proc:
            proc.stdout.close()
            proc.wait()
    finally:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass


@app.get("/api/tts-stream")
def tts_stream(text: str):
    # If ffmpeg is available we stream raw PCM (s16le, 16k mono) so frontend can
    # decode and schedule audio buffers for low-latency playback. If ffmpeg is not
    # available, the generator will return a WAV container and we must set the
    # content-type accordingly so the client can decode it via decodeAudioData or
    # by creating an Audio element from a Blob.
    media_type = "audio/pcm" if FFMPEG_AVAILABLE else "audio/wav"
    return StreamingResponse(
        fake_tts_stream(text),
        media_type=media_type
    )