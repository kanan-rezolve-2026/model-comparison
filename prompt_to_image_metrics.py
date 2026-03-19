import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "results/prompt-to-image"
OUTPUT_FILE = "metrics_results/prompt_to_image_metrics.txt"

PROMPT = """A cinematic portrait of a young man sitting in a dimly lit café at night, soft warm lighting from a table lamp illuminating his face, rain visible through the window behind him, shallow depth of field, reflections on glass, shot on a 50mm lens, ultra realistic skin texture, natural imperfections, detailed background, bokeh lights, 4K
"""

# -------- SETUP --------
os.makedirs("metrics_results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)

# -------- METRIC --------

def compute_clip(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([PROMPT]).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image_input)
        txt_feat = clip_model.encode_text(text_input)

        score = torch.cosine_similarity(img_feat, txt_feat).item()

        return max(min(score, 1.0), -1.0)

# -------- FILES --------

files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if len(files) == 0:
    raise ValueError("❌ No images found in folder!")

# -------- MAIN LOOP --------

with open(OUTPUT_FILE, "w") as f:
    f.write("model | file | CLIP | Aesthetic | Resolution\n")

    for file in tqdm(files):

        path = os.path.join(IMAGE_FOLDER, file)

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"❌ Skipping {file}: {e}")
            continue

        # ---- METRICS ----
        clip_score = compute_clip(img)
        aesthetic = (clip_score + 1) * 5

        # ---- RESOLUTION ----
        width, height = img.size
        resolution = f"{width}x{height}"

        model = os.path.splitext(file)[0]

        f.write(f"{model} | {file} | {clip_score:.4f} | {aesthetic:.2f} | {resolution}\n")

print("✅ Done Prompt→Image")