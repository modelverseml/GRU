"""
Run the whole project end to end with one command:

    python run_pipeline.py

It runs the three stages in order, each one depending on the last:

    1. data_generation.py            -> data/raw/{train,dev,test}.jsonl
    2. encoder.py                    -> data/text_encoder_small/  +  data/text_embedding/
    3. model_artifacts_generation.py -> data/model_artifacts/

Stage 2 downloads each pretrained encoder, trims it to our dataset's vocabulary
IN MEMORY (the full gigabyte files are never written to disk), saves the small
copy, then encodes the splits into .npz embeddings. BERT loads from HuggingFace.

If any stage fails the pipeline stops right there. Use the flags to skip
stages you've already done, e.g. once the data + encoders exist:

    python run_pipeline.py --skip data encoder
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# run every stage with the same interpreter that launched this script, so we
# stay inside the active virtualenv. all stages live next to this file.
CODE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# stage name -> (script, extra args). order here is the order they run in.
STAGES = [
    ("data",    ["data_generation.py"]),
    ("encoder", ["encoder.py"]),
    ("model",   ["model_artifacts_generation.py"]),
]


def run_stage(name, script_args, extra_args):
    # run one stage as a subprocess and stream its output live. returns nothing,
    # raises if the stage exits non-zero so the pipeline stops.
    cmd = [PYTHON, *script_args, *extra_args]
    print(f"\n{'=' * 70}")
    print(f">>> stage: {name}   ({' '.join(cmd)})")
    print(f"{'=' * 70}", flush=True)

    start = time.time()
    # cwd=CODE_DIR so the scripts' relative imports and paths resolve the same
    # way they do when you run them by hand from the code folder
    result = subprocess.run(cmd, cwd=CODE_DIR)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise SystemExit(
            f"\n!!! stage '{name}' failed (exit {result.returncode}) "
            f"after {elapsed:.1f}s -- pipeline stopped."
        )
    print(f"\n--- stage '{name}' done in {elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="run the full vanilla-RNN pipeline")
    # stages to skip (handy when raw data / encoders are already built)
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=[name for name, _ in STAGES],
        help="stage names to skip",
    )

    # ---- stage 1: data_generation.py ----
    p.add_argument(
        "--review-category-count", type=int, default=None,
        help="[data] how many product categories to pull",
    )
    p.add_argument(
        "--reviews-per-category", type=int, default=None,
        help="[data] reviews per category (split over the 5 ratings)",
    )

    # ---- stage 2: encoder.py ----
    p.add_argument(
        "--force-encoders", action="store_true",
        help="[encoder] re-download + re-trim the encoders even if present",
    )
    p.add_argument(
        "--encoder", default=None,
        choices=["all", "word2vec", "fasttext", "glove", "bert"],
        help="[encoder] which encoder(s) to build + encode",
    )
    p.add_argument(
        "--max-words", type=int, default=None,
        help="[encoder] words each title is padded/cut to",
    )

    return p.parse_args()


def stage_extra_args(name, args):
    # build the per-stage CLI args from the pipeline flags. only pass an arg
    # through when the user actually set it, so each script keeps its own default.
    extra = []
    if name == "data":
        if args.review_category_count is not None:
            extra += ["--review_category_count", str(args.review_category_count)]
        if args.reviews_per_category is not None:
            extra += ["--number_of_reviews_each_category", str(args.reviews_per_category)]
    elif name == "encoder":
        if args.force_encoders:
            extra += ["--force"]
        if args.encoder is not None:
            extra += ["--encoder", args.encoder]
        if args.max_words is not None:
            extra += ["--max_words", str(args.max_words)]
    # "model" stage takes no CLI args
    return extra


def main():
    args = parse_args()
    pipeline_start = time.time()

    for name, script_args in STAGES:
        if name in args.skip:
            print(f"\n>>> stage: {name}   (skipped)")
            continue
        run_stage(name, script_args, stage_extra_args(name, args))

    total = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"pipeline finished in {total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

