import torch
import collections
try:
    # allowlist common globals used in checkpoints
    torch.serialization.add_safe_globals([collections.defaultdict, dict])
    print("Added collections.defaultdict to torch safe globals")
except Exception as e:
    print("Warning: couldn't add collections.defaultdict to safe globals:", e)

try:
    # 先导入定义 RAdam 的模块，再把类加入 safe globals
    from TTS.utils.radam import RAdam
    torch.serialization.add_safe_globals([RAdam])
    # 或者（新 API）: torch.serialization.safe_globals([RAdam])
    print("Added RAdam to torch safe globals")
except Exception as e:
    print("Warning: couldn't add safe globals for RAdam:", e)

from TTS.api import TTS

tts = TTS(
    model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST",
    progress_bar=True
)

tts.tts_to_file(
    text="你好，这是一个完全本地运行的语音合成测试。",
    file_path="mytest.wav"
)