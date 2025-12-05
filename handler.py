import runpod
import sys

# ---------------------------------------------------------
# 起動直後のログ（これが表示されればPython自体は動いている）
# ---------------------------------------------------------
print("--- WORKER STARTED ---")

# ---------------------------------------------------------
# 1. 厳重なインポートチェック
# ---------------------------------------------------------
try:
    print("Loading standard libraries...")
    import base64
    import io
    import time
    
    print("Loading Torch...")
    import torch
    
    print("Loading Diffusers...")
    from diffusers import StableDiffusionInpaintPipeline
    
    print("Loading PIL...")
    from PIL import Image
    
    print("All imports successful!")

except ImportError as e:
    # ここで何が足りないかを表示して終了する
    print(f"!!! CRITICAL IMPORT ERROR !!!: {e}")
    sys.exit(1)
except Exception as e:
    print(f"!!! UNEXPECTED ERROR !!!: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. モデルのロード
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-inpainting"
pipe = None

print(f"Model Loading on {device}...")

try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None 
    ).to(device)
    # 高速化
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Model Load Failed: {e}")

# ---------------------------------------------------------
# 3. ヘルパー関数
# ---------------------------------------------------------
def decode_base64(string):
    return Image.open(io.BytesIO(base64.b64decode(string)))

def encode_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------------------------------------------------------
# 4. ハンドラ
# ---------------------------------------------------------
def handler(job):
    print(f"Received job: {job['id']}") # ログ用
    job_input = job["input"]
    
    image_str = job_input.get("image")
    mask_str = job_input.get("mask")
    prompt = job_input.get("prompt", "a cat")

    if not image_str or not mask_str:
        return {"error": "No image provided"}
    
    if pipe is None:
        return {"error": "Model not loaded"}

    try:
        init_image = decode_base64(image_str).convert("RGB").resize((512, 512))
        mask_image = decode_base64(mask_str).convert("RGB").resize((512, 512))

        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=20
        ).images[0]

        return {"image": encode_base64(output)}

    except Exception as e:
        print(f"Generation Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})