#!/usr/bin/env python3
"""Split a patient CSV into N chunks for parallel runs across machines.

Usage:
    python evaluate/split_csv.py --csv patients.csv --parts 6 --out-dir /tmp/chunks
    # produces: /tmp/chunks/part_01.csv ... part_06.csv

Each chunk is a valid CSV (with header) that can be passed to run_batch_aml.sh --csv.
"""
import argparse
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--parts", type=int, required=True, help="Number of chunks")
    ap.add_argument("--out-dir", default=".", help="Directory to write chunk CSVs")
    args = ap.parse_args()

    src = Path(args.csv)
    lines = src.read_text().splitlines()
    header = lines[0]
    patients = [l for l in lines[1:] if l.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = math.ceil(len(patients) / args.parts)
    digits = len(str(args.parts))

    for i in range(args.parts):
        chunk = patients[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            break
        out = out_dir / f"part_{i+1:0{digits}d}.csv"
        out.write_text(header + "\n" + "\n".join(chunk) + "\n")
        print(f"  {out}  ({len(chunk)} patients)")

    print(f"\nSplit {len(patients)} patients into {args.parts} chunks of ~{chunk_size}.")
    print("Run each chunk on a separate machine with:")
    print("  bash evaluate/run_batch_aml.sh --csv <chunk.csv> --resume ...")


if __name__ == "__main__":
    main()
