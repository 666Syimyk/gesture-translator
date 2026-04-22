import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "phrase_mode_target9_20260412"
)


def parse_csv(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def read_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest(source_dir, output_dir, unit_codes, split_samples):
    all_samples = []
    for samples in split_samples.values():
        all_samples.extend(samples)

    return {
        "source_dataset_dir": str(source_dir),
        "output_dataset_dir": str(output_dir),
        "unit_codes": unit_codes,
        "sample_count": len(all_samples),
        "split_counts": {
            split_name: len(samples) for split_name, samples in split_samples.items()
        },
        "coverage": dict(
            sorted(Counter(str(sample.get("phrase_text") or "").strip() for sample in all_samples).items())
        ),
        "unit_coverage": dict(
            sorted(Counter(str(sample.get("unit_code") or "").strip() for sample in all_samples).items())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit-codes", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    unit_codes = set(parse_csv(args.unit_codes))
    if not unit_codes:
        raise SystemExit("No unit codes provided.")

    split_samples = {}
    for split_name in ("train", "val", "test"):
        samples = read_json(source_dir / f"{split_name}.json")
        split_samples[split_name] = [
            sample
            for sample in samples
            if str(sample.get("unit_code") or "").strip() in unit_codes
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in split_samples.items():
        write_json(output_dir / f"{split_name}.json", samples)

    manifest = build_manifest(source_dir, output_dir, sorted(unit_codes), split_samples)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", manifest)


if __name__ == "__main__":
    main()
