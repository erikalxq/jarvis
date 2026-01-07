# jarvis
private AI agent

pip install uvicorn

安装python 3.11
https://www.python.org/downloads/release/python-3119/
勾选 Add Python to PATH

确认 Python 3.11
py -3.11 --version

用 Python 3.11 创建虚拟环境（非常关键）
cd G:\jarvis\server
py -3.11 -m venv venv311
venv311\Scripts\activate

重新安装依赖（在 venv311 里）
pip install fastapi uvicorn openai-whisper