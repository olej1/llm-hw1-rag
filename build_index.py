"""Сборка индекса: python build_index.py --corpus corpus.jsonl --output ./index --config config.json"""
import argparse
import json
import time

from retrieval import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    t0 = time.time()
    n = build(args.corpus, args.output, cfg)
    print(f"indexed {n} blocks -> {args.output} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
