from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import whisper
import tempfile
import os
import subprocess
import traceback
from pathlib import Path

app = FastAPI()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load model at startup (size configurable via WHISPER_MODEL env var)
model = None
MODEL_NAME = os.getenv('WHISPER_MODEL', 'small')


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
