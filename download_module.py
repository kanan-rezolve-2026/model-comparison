import requests
import os


def download_and_log(url, save_path, latency, model_name, txt_file_path):
    """
    url: CDN link
    save_path: full file path (with new name)
    latency: latency value
    model_name: model name
    txt_file_path: path of common txt file
    """

    # create folders if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    print(f"Downloading: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # -------- SAVE FILE (NO LOSS) --------
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Saved: {save_path}")

        # -------- APPEND LATENCY TO COMMON TXT --------
        with open(txt_file_path, "a") as f:
            f.write(f"{model_name} | {os.path.basename(save_path)} | {latency} sec\n")

        print(f"📊 Logged in: {txt_file_path}")

    except Exception as e:
        print(f"❌ Failed: {url}")
        print(e)


# -------- USAGE --------

download_and_log(
    url="",
    save_path="results/image-to-video/prompt-1/kling-v2-6-pro-freepik.mp4",  # 👈 SAVE AS MP4
    latency=11183.47  ,  # 👈 IN MILLISECONDS
    model_name="kling-v2-6-pro-freepik",  # 👈 MODEL NAME
    txt_file_path="C:/Kanan_Pandit_Rezolve_Ai/WORK-eko-eko/final-model-comp-19Mar/models_wise_data.txt"   # 👈 SAME FILE FOR ALL
)