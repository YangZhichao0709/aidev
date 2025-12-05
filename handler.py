import runpod
import torch
import base64
import io
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# ---------------------------------------------------------
# 1. モデルの設定とロード (Cold Start時に1回だけ実行)
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-inpainting"

print(f"モデルをロード中... ({device})")

try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # 安全性チェックを無効化する場合（真っ黒な画像が出るのを防ぐため）
        safety_checker=None 
    ).to(device)
    
    # 高速化のための設定
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        
    print("モデルロード完了！")
except Exception as e:
    print(f"モデルロードエラー: {e}")
    pipe = None

# ---------------------------------------------------------
# 2. ヘルパー関数
# ---------------------------------------------------------
def decode_base64(string):
    return Image.open(io.BytesIO(base64.b64decode(string)))

def encode_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------------------------------------------------------
# 3. ハンドラ関数 (リクエストのたびに実行)
# ---------------------------------------------------------
def handler(job):
    job_input = job["input"]
    
    # パラメータ取得
    image_str = job_input.get("image")
    mask_str = job_input.get("mask")
    prompt = job_input.get("prompt", "a cat sitting on a bench")
    negative_prompt = job_input.get("negative_prompt", "low quality, bad anatomy")
    steps = job_input.get("num_inference_steps", 25)
    guidance_scale = job_input.get("guidance_scale", 7.5)

    if not image_str or not mask_str:
        return {"error": "画像とマスクが必要です"}

    if pipe is None:
        return {"error": "モデルのロードに失敗しています"}

    try:
        # 画像変換
        init_image = decode_base64(image_str).convert("RGB").resize((512, 512))
        mask_image = decode_base64(mask_str).convert("RGB").resize((512, 512))

        # 推論実行
        # generator=torch.Generator(device).manual_seed(42) # 固定シードにしたい場合
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]

        # 結果返却
        return {"image": encode_base64(output)}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 4. サーバーレス起動
# ---------------------------------------------------------
runpod.serverless.start({"handler": handler})