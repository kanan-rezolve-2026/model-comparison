import os
import torch
import clip
import lpips
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

# -------- CONFIG --------
BASE_FOLDER = "results/edit/prompt-2"
OUTPUT_FILE = "metrics_results/edit_metrics_prompt-2.txt"

PROMPT = "Replace the background with a sunset beach scene with soft golden lighting. Match the lighting direction and color temperature on the subject so it looks natural."

os.makedirs("metrics_results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODELS --------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)


# -------- METRICS --------

def compute_clip(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([PROMPT]).to(device)

    with torch.no_grad():
        score = torch.cosine_similarity(
            clip_model.encode_image(image_input),
            clip_model.encode_text(text_input)
        ).item()

    return max(min(score, 1.0), -1.0)


def compute_lpips(img1, img2):
    # 🔥 FIX: resize both images to same size
    img1 = cv2.resize(np.array(img1), (256, 256))
    img2 = cv2.resize(np.array(img2), (256, 256))

    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255

    with torch.no_grad():
        return lpips_model(img1, img2).item()


def compute_ssim(img1, img2):
    img1 = cv2.resize(np.array(img1), (256, 256))
    img2 = cv2.resize(np.array(img2), (256, 256))
    return ssim(img1, img2, channel_axis=2)


def compute_psnr(img1, img2):
    img1 = cv2.resize(np.array(img1), (256, 256))
    img2 = cv2.resize(np.array(img2), (256, 256))
    mse = np.mean((img1 - img2) ** 2)

    return 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


# -------- LOAD REFERENCE --------

ref_path = os.path.join(BASE_FOLDER, "input_reference.png")

if not os.path.exists(ref_path):
    raise ValueError("❌ Reference image not found!")

ref_img = Image.open(ref_path).convert("RGB")

print("\n📂 Processing prompt-2")


# -------- MAIN LOOP --------

with open(OUTPUT_FILE, "w") as f:
    f.write("model | file | CLIP | SSIM | PSNR | LPIPS | Resolution\n")

    for file in tqdm(sorted(os.listdir(BASE_FOLDER))):

        # skip non-image + reference
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        if file == "input_reference.png":
            continue

        path = os.path.join(BASE_FOLDER, file)

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"❌ Skipping {file}: {e}")
            continue

        # ---- METRICS ----
        clip_score = compute_clip(img)
        ssim_score = compute_ssim(ref_img, img)
        psnr_score = compute_psnr(ref_img, img)
        lpips_score = compute_lpips(ref_img, img)

        # ---- RESOLUTION ----
        width, height = img.size
        resolution = f"{width}x{height}"

        model = os.path.splitext(file)[0]

        f.write(
            f"{model} | {file} | {clip_score:.4f} | {ssim_score:.4f} | {psnr_score:.2f} | {lpips_score:.4f} | {resolution}\n"
        )

print("\n✅ Done Edit Metrics (Prompt-2)")