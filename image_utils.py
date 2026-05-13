import base64
import io
import mimetypes
from typing import Dict, Any, List, Optional

from PIL import Image

from wsi_core_pkg.prompts import DEFAULT_AML_DIAGNOSIS_PROMPT
MAX_DIM = 1024
JPEG_QUALITY = 85


def encode_image_as_data_url(path: str, max_dim: int = MAX_DIM, jpeg_quality: int = JPEG_QUALITY) -> Optional[str]:
    """Load an image from disk, resize, JPEG-compress, and return a base64 data URI."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if max_dim > 0:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=max(20, min(95, jpeg_quality)), optimize=True)
            img_bytes = buf.getvalue()
            mime = "image/jpeg"
    except Exception:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            mime, _ = mimetypes.guess_type(path)
            if not mime:
                mime = "image/jpeg"
        except Exception:
            return None
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def make_image_part(path: str) -> Optional[Dict[str, Any]]:
    """Return an OpenAI-compatible image_url content part for a local image file."""
    url = encode_image_as_data_url(path)
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def build_encoded_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Convert a list of local image paths into OpenAI vision content parts."""
    parts = []
    for path in image_paths:
        part = make_image_part(path)
        if part:
            parts.append(part)
    return parts


# ── usage example ────────────────────────────────────────────────────────────
#
# from openai import OpenAI
# from image_utils import build_encoded_images
#
# client = OpenAI(base_url="http://pluto/v1", api_key="local")
#
# encoded_images = build_encoded_images(["slide1.jpg", "slide2.png"])
#
# response = client.chat.completions.create(
#     model="GLM-4.6V-FP8",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": [
#             {"type": "text", "text": user_prompt},
#             *encoded_images,
#         ]},
#     ],
# )


if __name__ == "__main__":
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="Image paths to encode")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://pluto/v1")
    parser.add_argument("--api-key", default="sk-y4X1YI9feTF_7KqflLuPPg")
    parser.add_argument("--prompt", default=DEFAULT_AML_DIAGNOSIS_PROMPT)
    args = parser.parse_args()

    if not args.images:
        print("Usage: python image_utils.py image1.jpg [--model gemma-4-31B-it]")
        sys.exit(0)

    # Step 1 — encode
    for path in args.images:
        print(f"\n--- {path} ---")
        if not os.path.exists(path):
            print("  ERROR: file not found")
            continue
        part = make_image_part(path)
        if part is None:
            print("  ERROR: failed to encode image")
            continue
        url = part["image_url"]["url"]
        print(f"  type       : {part['type']}")
        print(f"  url prefix : {url[:30]}...")
        print(f"  url length : {len(url)} chars")
        print("  encode: OK")

    # Step 2 — send to model if requested
    if args.model:
        from openai import OpenAI
        print(f"\n--- Sending to {args.model} at {args.base_url} ---")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        encoded = build_encoded_images([p for p in args.images if os.path.exists(p)])
        if not encoded:
            print("  ERROR: no valid images to send")
            sys.exit(1)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": args.prompt},
                    *encoded,
                ]},
            ],
        )
        print(f"  model reply: {response.choices[0].message.content}")

# EXP_NAME="gemma-4-31B-it_DinoBloom-G_224px"

# bash evaluate/run_batch_aml_diagnosis.sh \
#   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis/${EXP_NAME}" \
#   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis" \
#   --model "gemma-4-31B-it" \
#   --chunks-dir "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/chunks4" \
#   --chunk-index 3 \
#   --resume

#   bash evaluate/run_batch_aml_diagnosis.sh   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_290425_aml_suite"   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis_090525"   --model "gemma-4-31B-it"   --subdir-filter "gemma-4-31B-it_UNI2_224px,gemma-4-31B-it_Virchow2_224px,gemma-4-31B-it_H-optimus-1_224px,gemma-4-31B-it_DinoBloom-G_224px"   --parallel   --resume

# bash /mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent/evaluate/run_batch_aml_diagnosis.sh \
#   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis" \
#   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/aml_diagnosis_100525" \
#   --model "Qwen3.5-397B-A17B-FP8" \
#   --subdir-filter "Qwen3.5-397B-A17B-FP8_UNI2_224px" \
#   --chunks-dir "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/chunks" \
#   --chunk-index 1 \
#   --resume


# bash /mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent/evaluate/run_batch_aml_diagnosis.sh \
#   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_290425_aml_suite" \
#   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/aml_diagnosis_100525_3" \
#   --model "GLM-4.6V-FP8" \
#   --subdir-filter "GLM-4.6V-FP8_UNI2_224px,GLM-4.6V-FP8_H-optimus-1_224px,GLM-4.6V-FP8_Virchow2_224px,GLM-4.6V-FP8_DinoBloom-G_224px" \
#   --parallel \
#   --resume

#   bash /mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent/evaluate/run_batch_aml_diagnosis.sh \
#   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis" \
#   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/aml_diagnosis_100525_1" \
#   --model "Qwen3.5-397B-A17B-FP8" \
#   --subdir-filter "Qwen3.5-397B-A17B-FP8_UNI2_224px,Qwen3.5-397B-A17B-FP8_H-optimus-1_224px,Qwen3.5-397B-A17B-FP8_Virchow2_224px,Qwen3.5-397B-A17B-FP8_DinoBloom-G_224px" \
#   --parallel \
#   --resume

#   tmux new-session -d -s qwen 'bash /mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent/evaluate/run_batch_aml_diagnosis.sh \
#   --exp-path "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis" \
#   --output-root "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/aml_diagnosis_100525_1" \
#   --model "Qwen3.5-397B-A17B-FP8" \
#   --subdir-filter "Qwen3.5-397B-A17B-FP8_UNI2_224px,Qwen3.5-397B-A17B-FP8_H-optimus-1_224px,Qwen3.5-397B-A17B-FP8_Virchow2_224px,Qwen3.5-397B-A17B-FP8_DinoBloom-G_224px" \
#   --parallel \
#   --resume'


# CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
# SLIDES="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
# OUT_ROOT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_diagnosis"
# SCRIPT="/mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent/evaluate/run_batch_aml.sh"

# for ext in uni2 virchow2 h_optimus_1 dinobloom_giant; do
#   case "$ext" in
#     dinobloom_giant) tag="DinoBloom-G" ;;
#   esac

#   bash "$SCRIPT" \
#     --csv "$CSV" \
#     --slides-root "$SLIDES" \
#     --output-dir "${OUT_ROOT}/gemma-4-31B-it_${tag}_224px" \
#     --experiment-root "$OUT_ROOT" \
#     --model "gemma-4-31B-it" \
#     --extractor "$ext" \
#     --tile-filter hybrid \
#     --tile-size-px 224 \
#     --batch-size 512 \
#     --roi-size-px 2048 \
#     --default-mpp-um 0.159 \
#     --agent aml_roi \
#     --resume
# done