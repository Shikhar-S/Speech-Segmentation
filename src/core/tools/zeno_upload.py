"""Upload predictions to Zeno from JSONL file.
Usage:
    python -m src.core.tools.zeno_upload \
        --input /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/xeusresultsdump/buckeye.jsonl \
        --api-key zen_G7yVGaqFT6C69mT9XequKW9sOyIOis3EqCBUlJT7oMA \
        --project-name buckeye \
        --model-name phonetic-xeus \
        --s3-base-url https://l2arctic.s3.us-east-2.amazonaws.com
        
aws s3 sync /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/exp/cache/xeusresultsdump \
    s3://l2arctic/ --exclude "*" --include "*.wav"
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from zeno_client import ZenoClient, ZenoMetric


def main(args):
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    # data = data[:200] # test
    base_df = pd.DataFrame(
        {
            "id": [d.get("key", d.get("id", Path(d["wavpath"]).stem)) for d in data],
            "data": [f"{args.s3_base_url}/{Path(d['wavpath']).name}" for d in data],
            # label becomes a dict so we can show multiple fields in the label panel
            "label": [
                {
                    "label": d.get("phone_str", ""),  # what you used to call "label"
                    "asr_text": d.get("asr_text", ""),  # extra field you want displayed
                }
                for d in data
            ],
            "language": [d.get("language", "") for d in data],
        }
    )

    # base_df = pd.DataFrame(
    #     {
    #         "id": [d.get("key", d.get("id", Path(d["wavpath"]).stem)) for d in data],
    #         "data": [f"{args.s3_base_url}/{Path(d['wavpath']).name}" for d in data],
    #         "asr_text": [d.get("asr_text", "") for d in data],
    #         "label": [d.get("phone_str", "") for d in data],
    #         "language": [d.get("language", "") for d in data],
    #     }
    # )

    metric_keys = list(data[0].get("pr_metrics", {}).keys())
    if "inventory" in metric_keys:
        metric_keys.remove("inventory")
    model_df = pd.DataFrame(
        {
            "id": base_df["id"],
            "output": [d.get("prediction", "") for d in data],
            "loss": [d.get("loss", 0.0) for d in data],
            **{col: [d["pr_metrics"][col] for d in data] for col in metric_keys},
        }
    )

    # loss

    client = ZenoClient(args.api_key)
    view = {
        "data": {"type": "audio"},
        "label": {
            "type": "vstack",
            "keys": {
                "label": {"type": "text", "label": "Label"},
                "asr_text": {"type": "text", "label": "ASR Text"},
            },
        },
        "output": {"type": "text", "label": "Output"},
        "size": "large",
    }
    # view = "audio-transcription"
    project = client.create_project(
        name=args.project_name,
        view=view,
        metrics=[
            ZenoMetric(name=col, type="mean", columns=[col])
            for col in metric_keys + ["loss"]
        ],
    )
    project.upload_dataset(
        base_df, id_column="id", data_column="data", label_column="label"
    )
    project.upload_system(
        model_df, name=args.model_name, id_column="id", output_column="output"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--api-key", required=True, help="Zeno API key")
    parser.add_argument("--project-name", required=True, help="Zeno project name")
    parser.add_argument(
        "--s3-base-url", required=True, help="S3 base URL for audio files"
    )
    parser.add_argument(
        "--model-name", default="model", help="Model name for the system"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
