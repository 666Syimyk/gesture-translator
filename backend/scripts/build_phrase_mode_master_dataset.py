import argparse
import json
from collections import Counter
from pathlib import Path

from bootstrap_phrase_pack_signs import build_slovo_samples
from bootstrap_slovo_signs import (
    DEFAULT_ANNOTATIONS_URL,
    DEFAULT_CACHE_DIR as DEFAULT_SLOVO_CACHE_DIR,
    DEFAULT_CONSTANTS_URL,
    DEFAULT_LANDMARKS_URL,
    extract_annotations_csv_from_zip,
    load_annotations,
    load_slovo_constants,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_DATASET_DIR = REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "user_words_20260407"
DEFAULT_NEW_DATASET_DIR = REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "phrase_pack_20260412"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "phrase_mode_master_20260412"

OLD_TARGET_LABELS = {
    "\u0414\u0430",
    "\u041d\u0435\u0442",
    "\u041f\u0440\u0438\u0432\u0435\u0442",
    "\u041f\u043e\u043a\u0430",
    "\u0414\u0440\u0443\u0433",
}

NEW_TARGET_LABELS = {
    "\u041c\u0443\u0436\u0447\u0438\u043d\u0430",
    "\u0416\u0435\u043d\u0449\u0438\u043d\u0430",
    "\u0421\u043e\u043b\u043d\u0446\u0435",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_filtered_split(dataset_dir, target_labels):
    result = {}
    for split in ("train", "val", "test"):
        samples = read_json(dataset_dir / f"{split}.json")
        result[split] = [
            sample for sample in samples if sample.get("phrase_text") in target_labels
        ]
    return result


def write_dataset(output_dir, train_samples, val_samples, test_samples):
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        (output_dir / f"{split}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_samples = train_samples + val_samples + test_samples
    manifest = {
        "source_dataset": "Phrase mode master pack",
        "recognition_level": "sign",
        "feature_mode": "mixed_hands_and_holistic",
        "sample_count": len(all_samples),
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "class_count": len({sample["phrase_text"] for sample in all_samples}),
        "coverage": dict(sorted(Counter(sample["phrase_text"] for sample in all_samples).items())),
        "train_coverage": dict(sorted(Counter(sample["phrase_text"] for sample in train_samples).items())),
        "val_coverage": dict(sorted(Counter(sample["phrase_text"] for sample in val_samples).items())),
        "test_coverage": dict(sorted(Counter(sample["phrase_text"] for sample in test_samples).items())),
        "external_sources": {
            "slovo": "https://github.com/hukenovs/slovo",
            "signflow": "https://signflow.ru/phrases",
        },
        "notes": (
            "\u041e\u0431\u0449\u0438\u0439 phrase-mode pack \u0434\u043b\u044f 9 \u0446\u0435\u043b\u0435\u0432\u044b\u0445 "
            "\u0441\u043b\u043e\u0432. \u0421\u0442\u0430\u0440\u044b\u0435 5 \u0441\u043b\u043e\u0432 \u0431\u0435\u0440\u0443\u0442\u0441\u044f "
            "\u0438\u0437 user-words dataset, \u0414\u043e\u043c \u0438\u0437 Slovo, \u041c\u0443\u0436\u0447\u0438\u043d\u0430/\u0416\u0435\u043d\u0449\u0438\u043d\u0430/"
            "\u0421\u043e\u043b\u043d\u0446\u0435 \u0438\u0437 phrase-pack booster dataset."
        ),
    }

    for name in ("manifest.json", "summary.json"):
        (output_dir / name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dataset-dir", default=str(DEFAULT_OLD_DATASET_DIR))
    parser.add_argument("--new-dataset-dir", default=str(DEFAULT_NEW_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--slovo-cache-dir", default=str(DEFAULT_SLOVO_CACHE_DIR))
    parser.add_argument("--annotations-zip-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--landmarks-url", default=DEFAULT_LANDMARKS_URL)
    parser.add_argument("--constants-url", default=DEFAULT_CONSTANTS_URL)
    args = parser.parse_args()

    old_dataset_dir = Path(args.old_dataset_dir)
    new_dataset_dir = Path(args.new_dataset_dir)
    output_dir = Path(args.output_dir)
    slovo_cache_dir = Path(args.slovo_cache_dir)

    old_samples = load_filtered_split(old_dataset_dir, OLD_TARGET_LABELS)
    new_samples = load_filtered_split(new_dataset_dir, NEW_TARGET_LABELS)

    annotations_path = extract_annotations_csv_from_zip(
        args.annotations_zip_url,
        slovo_cache_dir / "annotations.csv",
    )
    annotations = load_annotations(annotations_path)
    constants = load_slovo_constants(args.constants_url)
    dom_spec = {
        "display_text": "\u0414\u043e\u043c",
        "unit_code": "SIGN_PHRASEPACK_DOM",
        "slovo_labels": ["\u0434\u043e\u043c"],
    }
    dom_samples = build_slovo_samples(
        dom_spec,
        annotations=annotations,
        constants=constants,
        cache_dir=slovo_cache_dir,
        landmarks_url=args.landmarks_url,
    )

    combined = {}
    for split in ("train", "val", "test"):
        combined[split] = [
            *old_samples[split],
            *dom_samples[split],
            *new_samples[split],
        ]

    write_dataset(
        output_dir,
        train_samples=combined["train"],
        val_samples=combined["val"],
        test_samples=combined["test"],
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "splitCounts": {
                    split: len(samples) for split, samples in combined.items()
                },
                "trainCoverage": dict(
                    sorted(Counter(sample["phrase_text"] for sample in combined["train"]).items())
                ),
                "valCoverage": dict(
                    sorted(Counter(sample["phrase_text"] for sample in combined["val"]).items())
                ),
                "testCoverage": dict(
                    sorted(Counter(sample["phrase_text"] for sample in combined["test"]).items())
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
