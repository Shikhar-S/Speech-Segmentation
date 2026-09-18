"""Push the 5 prepared Global TIMIT segmentation datasets to the HuggingFace Hub.

Local DatasetDicts (saved by ``scripts/data_prep/gtimit_data_prep.py``)
live under ``exp/data/global-timit-*/``. This script:

1. Loads each one with ``load_from_disk``.
2. Calls ``push_to_hub`` (creates the repo if it doesn't exist, uploads
   parquet shards, and writes the standard Hub layout).
3. Uploads the local ``README.md`` as the dataset card so the speaker /
   accent distribution tables and frontmatter render on the Hub page.

Usage:
    huggingface-cli login                        # one time
    python scripts/data_prep/push_gtimit_to_hub.py --org <your-org> [--public]

Repos are named ``<your-org>/gtimit-<subset>-segment``; sub-corpora whose local
directory is missing are skipped, so preparing only the Table I subsets
(L2ENGsimple, L2ENGtreebank, THA) is fine. Point ``SEG_REPO_GTIMIT_*`` at the
resulting ids.
"""

import argparse
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


REPO_NAMES = {
    "exp/data/global-timit-l1eng-simple-segment": "gtimit-l1simple-segment",
    "exp/data/global-timit-l2eng-simple-segment": "gtimit-l2simple-segment",
    "exp/data/global-timit-l1eng-tbnk-segment":   "gtimit-l1tbnk-segment",
    "exp/data/global-timit-l2eng-tbnk-segment":   "gtimit-l2tbnk-segment",
    "exp/data/global-timit-tha-segment":          "gtimit-tha-segment",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--org", required=True, help="HF user or org (e.g. changelinglab).")
    p.add_argument(
        "--public",
        action="store_true",
        help="Push as public. Default is private (LDC license safer).",
    )
    p.add_argument(
        "--readme-only",
        action="store_true",
        help="Only re-upload README.md (skip data push). Useful for card edits.",
    )
    args = p.parse_args()

    api = HfApi()
    private = not args.public

    for local, repo_short in REPO_NAMES.items():
        repo_id = f"{args.org}/{repo_short}"
        local_path = Path(local)
        readme_path = local_path / "README.md"
        if not local_path.is_dir():
            print(f"skip {repo_id}: {local_path} not prepared")
            continue

        print(f"\n=== {local_path} -> {repo_id} (private={private}) ===")
        if not args.readme_only:
            ds = load_from_disk(str(local_path))
            ds.push_to_hub(repo_id, private=private)

        if not readme_path.is_file():
            print(f"OK: pushed data for {repo_id} (no local README.md to upload)")
            continue
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card with speaker/accent distribution",
        )
        print(f"OK: pushed data and uploaded README for {repo_id}")

    print("\nAll 5 datasets pushed.")


if __name__ == "__main__":
    main()
