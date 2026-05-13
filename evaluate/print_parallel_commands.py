#!/usr/bin/env python3
"""Print ready-to-run batch commands for each machine+chunk combination.

Usage:
    python evaluate/print_parallel_commands.py \
        --chunks-dir /tmp/chunks \
        --machines 6 \
        --suite Qwen3.5-397B-A17B-FP8 \
        --extractors uni2 virchow2 h_optimus_1 dinobloom_giant \
        --experiment-root /mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_rerun_clean

Prints one command per line, assign one line per machine terminal.
Each command processes one CSV chunk for all extractors sequentially on that machine.
"""
import argparse
from pathlib import Path


EXTRACTOR_TAGS = {
    "uni2": "UNI2",
    "h_optimus_1": "H-optimus-1",
    "virchow2": "Virchow2",
    "dinobloom_giant": "DinoBloom-G",
    "dinobloom": "DinoBloom-S",
}

SLIDES_ROOT = "/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
REPO = "/mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--extractors", nargs="+",
                    default=["uni2", "virchow2", "h_optimus_1", "dinobloom_giant"])
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--tile-filter", default="hybrid")
    ap.add_argument("--tile-size-px", default="224")
    ap.add_argument("--batch-size", default="512")
    ap.add_argument("--roi-size-px", default="2048")
    args = ap.parse_args()

    chunks = sorted(Path(args.chunks_dir).glob("part_*.csv"))
    if not chunks:
        print(f"No part_*.csv found in {args.chunks_dir}")
        return

    print(f"# {len(chunks)} chunks × {len(args.extractors)} extractors")
    print(f"# Model: {args.model}")
    print("# Assign one block per machine\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"# ── Machine {i} — {chunk.name} ({'–'.join(args.extractors)}) ──")
        for ext in args.extractors:
            tag = EXTRACTOR_TAGS.get(ext, ext)
            out_dir = f"{args.experiment_root}/{args.model}_{tag}_{args.tile_size_px}px"
            cmd = (
                f"cd {REPO} && "
                f"bash evaluate/run_batch_aml.sh"
                f" --csv {chunk}"
                f" --slides-root {SLIDES_ROOT}"
                f" --output-dir {out_dir}"
                f" --experiment-root {args.experiment_root}"
                f" --model {args.model}"
                f" --extractor {ext}"
                f" --tile-filter {args.tile_filter}"
                f" --tile-size-px {args.tile_size_px}"
                f" --batch-size {args.batch_size}"
                f" --roi-size-px {args.roi_size_px}"
                f" --resume"
                f" --use-tile-cache"
            )
            print(cmd)
        print()


if __name__ == "__main__":
    main()
