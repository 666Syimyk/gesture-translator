import argparse
import json
import sys
from pathlib import Path

from predict_sequence_model import build_prediction_output, load_model_bundle


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    bundle = load_model_bundle(Path(args.model_dir))

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None

        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            result = build_prediction_output(
                bundle,
                input_path=payload.get("inputPath"),
                sequence=payload.get("sequence"),
                allowed_levels=payload.get("allowedRecognitionLevels") or [],
                allowed_label_keys=payload.get("allowedLabelKeys") or [],
            )
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "result": result,
                }
            )
        except Exception as error:
            emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(error),
                }
            )


if __name__ == "__main__":
    main()
