import argparse
import json
import re
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path

from bootstrap_phrase_pack_signs import (
    LABEL_SPECS as BASE_SIGNFLOW_LABEL_SPECS,
    build_signflow_variants,
    build_slovo_samples,
)
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
DEFAULT_OLD_DATASET_DIR = (
    REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "user_words_signflow_allseed_20260407"
)
DEFAULT_SIGNFLOW_DIR = REPO_ROOT / "backend" / "uploads" / "external" / "signflow"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "phrase_mode_target9_20260412"
)

CORE_OLD_LABELS = {
    "Да": "SIGN_USER_DA",
    "Нет": "SIGN_USER_NET",
    "Пока": "SIGN_USER_POKA",
    "Привет": "SIGN_USER_PRIVET",
}
PROXY_OLD_LABELS = {"Друг"}

SPECIAL_SPECS = {
    "Дом": {
        "unit_code": "SIGN_PHRASEPACK_DOM",
        "slug": "dom",
        "signflow_page": "https://signflow.ru/dom",
        "signflow_video": None,
        "slovo_labels": ["дом"],
    },
    "Дружба": {
        "unit_code": "SIGN_PHRASEPACK_DRUZHBA",
        "slug": "druzhba",
        "signflow_page": "https://signflow.ru/druzhba",
        "signflow_video": None,
        "slovo_labels": [],
        "proxy_from_old_label": "Друг",
    },
    "Мужчина": {
        "unit_code": "SIGN_PHRASEPACK_MUZHCHINA",
        "slug": "muzhchina",
        "signflow_page": "https://signflow.ru/muzhchina",
        "signflow_video": None,
        "slovo_labels": ["мужчина"],
    },
    "Женщина": {
        "unit_code": "SIGN_PHRASEPACK_ZHENSHCHINA",
        "slug": "zhenshchina",
        "signflow_page": "https://signflow.ru/zhenshchina",
        "signflow_video": None,
        "slovo_labels": ["женщина"],
    },
    "Солнце": {
        "unit_code": "SIGN_PHRASEPACK_SOLNTSE",
        "slug": "solntse",
        "signflow_page": "https://signflow.ru/solntse",
        "signflow_video": None,
        "slovo_labels": ["солнце"],
    },
}


def read_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def fetch_signflow_video_url(page_url):
    with urllib.request.urlopen(page_url, timeout=60) as response:
        html = response.read().decode("utf-8", "ignore")

    match = re.search(r"https://static\.signflow\.ru/p-rsl-data/video/[^\"'\s]+\.mp4", html)
    return match.group(0) if match else None


def load_filtered_split(dataset_dir, target_labels):
    result = {}
    for split in ("train", "val", "test"):
        samples = read_json(dataset_dir / f"{split}.json")
        result[split] = [
            sample
            for sample in samples
            if str(sample.get("phrase_text") or "").strip() in target_labels
        ]
    return result


def remap_sample(sample, display_text, unit_code, source_note=None):
    next_sample = deepcopy(sample)
    original_text = str(next_sample.get("phrase_text") or "").strip()

    next_sample["sample_id"] = f"target9:{display_text.lower()}:{next_sample.get('sample_id')}"
    next_sample["label_id"] = unit_code
    next_sample["phrase_id"] = unit_code
    next_sample["phrase_text"] = display_text
    next_sample["unit_code"] = unit_code

    sequence_meta = dict(next_sample.get("sequence_meta") or {})
    sequence_meta["target_text"] = display_text
    sequence_meta["text"] = display_text
    if original_text and original_text != display_text:
        sequence_meta["source_text"] = original_text
    if source_note:
        sequence_meta["source_note"] = source_note
    next_sample["sequence_meta"] = sequence_meta

    return next_sample


def write_dataset(output_dir, train_samples, val_samples, test_samples, source_manifest):
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
        "source_dataset": "Phrase mode target9 pack",
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
        "external_sources": source_manifest,
        "notes": (
            "Точный pack для целевых 9 фраз. Да/Нет/Привет/Пока берутся из user_words_signflow_allseed, "
            "Дом/Мужчина/Женщина/Солнце усиливаются Slovo + SignFlow, "
            "Дружба собирается из точного SignFlow seed и proxy-примеров Друг."
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
    parser.add_argument("--signflow-dir", default=str(DEFAULT_SIGNFLOW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--slovo-cache-dir", default=str(DEFAULT_SLOVO_CACHE_DIR))
    parser.add_argument("--annotations-zip-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--landmarks-url", default=DEFAULT_LANDMARKS_URL)
    parser.add_argument("--constants-url", default=DEFAULT_CONSTANTS_URL)
    args = parser.parse_args()

    old_dataset_dir = Path(args.old_dataset_dir)
    signflow_dir = Path(args.signflow_dir)
    output_dir = Path(args.output_dir)
    slovo_cache_dir = Path(args.slovo_cache_dir)

    old_samples = load_filtered_split(old_dataset_dir, set(CORE_OLD_LABELS) | PROXY_OLD_LABELS)

    annotations_path = extract_annotations_csv_from_zip(
        args.annotations_zip_url,
        slovo_cache_dir / "annotations.csv",
    )
    annotations = load_annotations(annotations_path)
    constants = load_slovo_constants(args.constants_url)

    buckets = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        buckets[split].extend(
            [
                sample
                for sample in old_samples[split]
                if str(sample.get("phrase_text") or "").strip() in CORE_OLD_LABELS
            ]
        )

    source_manifest = {
        "slovo": "https://github.com/hukenovs/slovo",
        "signflow": "https://signflow.ru/phrases",
        "signflow_words": {},
    }

    for display_text, raw_spec in SPECIAL_SPECS.items():
        spec = {**raw_spec, "display_text": display_text}
        spec["signflow_video"] = fetch_signflow_video_url(spec["signflow_page"])
        BASE_SIGNFLOW_LABEL_SPECS[display_text] = spec
        if spec["signflow_video"]:
            source_manifest["signflow_words"][display_text] = spec["signflow_page"]

        signflow_samples = (
            build_signflow_variants(spec, signflow_dir)
            if spec["signflow_video"]
            else {"train": [], "val": [], "test": []}
        )
        slovo_samples = build_slovo_samples(
            spec,
            annotations=annotations,
            constants=constants,
            cache_dir=slovo_cache_dir,
            landmarks_url=args.landmarks_url,
        )

        for split in ("train", "val", "test"):
            buckets[split].extend(signflow_samples[split])
            buckets[split].extend(slovo_samples[split])

        proxy_label = spec.get("proxy_from_old_label")
        if proxy_label:
            proxy_note = (
                "Proxy-пример: исходный жест был размечен как Друг и переиспользован "
                "для усиления целевой фразы Дружба."
            )
            for split in ("train", "val", "test"):
                proxy_samples = [
                    remap_sample(
                        sample,
                        display_text=display_text,
                        unit_code=spec["unit_code"],
                        source_note=proxy_note,
                    )
                    for sample in old_samples[split]
                    if str(sample.get("phrase_text") or "").strip() == proxy_label
                ]
                buckets[split].extend(proxy_samples)

    write_dataset(
        output_dir,
        train_samples=buckets["train"],
        val_samples=buckets["val"],
        test_samples=buckets["test"],
        source_manifest=source_manifest,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "splitCounts": {split: len(samples) for split, samples in buckets.items()},
                "trainCoverage": dict(sorted(Counter(sample["phrase_text"] for sample in buckets["train"]).items())),
                "valCoverage": dict(sorted(Counter(sample["phrase_text"] for sample in buckets["val"]).items())),
                "testCoverage": dict(sorted(Counter(sample["phrase_text"] for sample in buckets["test"]).items())),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
