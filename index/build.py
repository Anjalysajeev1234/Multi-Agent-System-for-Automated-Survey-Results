# index/build.py
"""
Minimal placeholder for the index builder.

Keeps the CLI contract only (arguments parsing). No indexing logic.
Usage:
    python -m index.build --config config.yaml --api_key <YOUR_KEY>
"""

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="(placeholder) Build indexes from documents.jsonl"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Together (or other) API key; accepted but unused in placeholder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Placeholder: just acknowledge arguments.
    print(f"[index.build] config={args.config}  api_key={'<provided>' if args.api_key else '<none>'}")
    print("[index.build] TODO: implement indexing logic here.")


if __name__ == "__main__":
    main()
