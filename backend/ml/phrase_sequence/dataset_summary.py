import argparse
import json
import sys
from pathlib import Path


PHRASE_DIR = Path(__file__).resolve().parent
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))

from config import load_config, resolve_backend_path
from data import SPLITS, build_dataset_summary


def print_table(summary):
    labels = summary["labels"]
    print(f"Dataset: {summary['dataset_dir']}")
    print("label".ljust(22) + "train".rjust(10) + "val".rjust(10) + "test".rjust(10))
    print("-" * 52)

    for label in labels:
        cells = [label.ljust(22)]
        for split in SPLITS:
            item = summary["splits"][split][label]
            value = f"{item['usable_count']}/{item['count']}"
            cells.append(value.rjust(10))
        print("".join(cells))

    print("-" * 52)
    totals = ["TOTAL".ljust(22)]
    for split in SPLITS:
        totals.append(str(summary["totals"][split]).rjust(10))
    print("".join(totals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = resolve_backend_path(args.dataset_dir or config["dataset_dir"])
    summary = build_dataset_summary(
        dataset_dir,
        config["phrases"],
        quality_config=config.get("quality", {}),
        idle_label=config["inference"].get("idle_label", "none"),
    )

    if args.json:
        print(json.dumps({"ok": True, "summary": summary}, ensure_ascii=False, indent=2))
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
