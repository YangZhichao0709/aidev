import runpod
import torch
import base64
import io
from PIL import Image

# あなたの既存コードからモデル読み込みと推論関数をインポート
# 例: from my_model_file import load_model, inpaint_function
# 仮置き:
def load_model():
    print("Dummy Model Loaded")
    return "MODEL"
def inpaint_function(model, img, mask, prompt):
    return img # そのまま返すダミー

print("モデルロード開始...")
pipe = load_model()
print("モデルロード完了")

def decode_base64(string):
    return Image.open(io.BytesIO(base64.b64decode(string)))

def encode_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    job_input = job["input"]
    
    # 入力取得
    image_str = job_input.get("image")
    mask_str = job_input.get("mask")
    prompt = job_input.get("prompt", "")

    if not image_str or not mask_str:
        return {"error": "No image or mask provided"}

    # 処理
    image = decode_base64(image_str)
    mask = decode_base64(mask_str)
    
    # 推論実行
    result_image = inpaint_function(pipe, image, mask, prompt)
    
    # 返却
    return {"image": encode_base64(result_image)}

runpod.serverless.start({"handler": handler})