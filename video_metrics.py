import os
import cv2
import torch
import clip
import numpy as np
import lpips
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# -------- CONFIG --------
BASE_FOLDER = "results/image-to-video/prompt-1"
OUTPUT_FILE = "metrics_results/video_metrics_prompt-1.txt"
REF_IMAGE_PATH = os.path.join(BASE_FOLDER, "input_reference.png")

PROMPT = "A woman sitting at a café, then standing up and walking away, continuous shot, consistent face and outfit, smooth transition, cinematic lighting"

os.makedirs("metrics_results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODELS --------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
lpips_model = lpips.LPIPS(net='alex').to(device)

# -------- LOAD REFERENCE IMAGE --------
if not os.path.exists(REF_IMAGE_PATH):
    raise ValueError("❌ input_reference.png not found!")

ref_img = Image.open(REF_IMAGE_PATH).convert("RGB")


# -------- FUNCTIONS --------

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return frames

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / num_frames))
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def compute_clip(frames):
    if len(frames) == 0:
        return 0

    scores = []
    text = clip.tokenize([PROMPT]).to(device)

    for img in frames:
        img_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = torch.cosine_similarity(
                clip_model.encode_image(img_input),
                clip_model.encode_text(text)
            ).item()

        score = max(min(score, 1.0), -1.0)
        scores.append(score)

    return sum(scores) / len(scores)


def compute_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            diffs.append(np.mean(cv2.absdiff(prev, gray)))

        prev = gray

    cap.release()

    return sum(diffs) / len(diffs) if len(diffs) > 0 else 0


def compute_ssim(img1, img2):
    img1 = cv2.resize(np.array(img1), (256, 256))
    img2 = cv2.resize(np.array(img2), (256, 256))
    return ssim(img1, img2, channel_axis=2)


def compute_lpips(img1, img2):
    img1 = cv2.resize(np.array(img1), (256, 256))
    img2 = cv2.resize(np.array(img2), (256, 256))

    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255

    with torch.no_grad():
        return lpips_model(img1, img2).item()


# -------- MAIN --------

print("\n📂 Processing Video Prompt-1")

with open(OUTPUT_FILE, "w") as f:
    f.write("model | file | CLIP | Motion | Consistency | FirstFrame_SSIM | FirstFrame_LPIPS\n")

    for file in tqdm(sorted(os.listdir(BASE_FOLDER))):

        if not file.lower().endswith(".mp4"):
            continue

        path = os.path.join(BASE_FOLDER, file)

        # ---- FRAME EXTRACTION ----
        frames = extract_frames(path)
        first_frame = get_first_frame(path)

        if first_frame is None:
            print(f"❌ Skipping {file} (no frames)")
            continue

        # ---- METRICS ----
        clip_score = compute_clip(frames)
        motion = compute_motion(path)
        consistency = 1 / (1 + motion)

        # ---- FIRST FRAME COMPARISON ----
        ssim_score = compute_ssim(ref_img, first_frame)
        lpips_score = compute_lpips(ref_img, first_frame)

        model = os.path.splitext(file)[0]

        f.write(
            f"{model} | {file} | {clip_score:.4f} | {motion:.2f} | {consistency:.4f} | {ssim_score:.4f} | {lpips_score:.4f}\n"
        )

print("\n✅ Done Video Metrics (Prompt-1)")