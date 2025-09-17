# mas_survey/run.py
"""
Minimal placeholder for the MAS runner (two arguments only).

CLI:
    python -m mas_survey.run --config config.yaml --api_key <YOUR_KEY>
"""

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(placeholder) Run MAS pipeline")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--api_key", type=str, default=None,
                   help="API key (accepted but unused in placeholder).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[mas_survey.run] config={args.config}  api_key={'<provided>' if args.api_key else '<none>'}")
    print("[mas_survey.run] TODO: implement pipeline and write artifacts/submission_js.csv")


if __name__ == "__main__":
    main()
