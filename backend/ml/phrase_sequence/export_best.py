import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PHRASE_DIR = Path(__file__).resolve().parent
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))

from config import load_config, resolve_backend_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir", default="ml/artifacts/phrase_sequence/best")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source_dir = resolve_backend_path(args.model_dir or config["artifacts_dir"])
    output_dir = resolve_backend_path(args.output_dir)

    if not (source_dir / "metadata.json").exists():
        raise SystemExit(f"No trained model metadata found in {source_dir}.")

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_dir} already exists. Re-run with --overwrite.")
        shutil.rmtree(output_dir)

    shutil.copytree(source_dir, output_dir)
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "purpose": "best_phrase_sequence_model_export",
    }
    with (output_dir / "export_manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()

