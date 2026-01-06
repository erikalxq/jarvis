from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import whisper
import tempfile

app = FastAPI()
model = whisper.load_model(
    "base",
    download_root="./models"
)

# 前端
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # print logging info
    print(f"Received file: {audio.filename}, content type: {audio.content_type}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    
    print(f"Saved temporary file at: {tmp_path}")
    result = model.transcribe(tmp_path, language="zh")
    return {"text": result["text"]}
