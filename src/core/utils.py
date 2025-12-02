from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
import logging


def download_hf_snapshot(
    repo_id: str,
    work_dir: str,
    revision: str = None,
    force_download: bool = False,
    **kwargs,
) -> str:
    """
    Download a snapshot from Hugging Face Hub to `work_dir`, but skip network interaction if
    the snapshot already exists locally.

    Args:
        repo_id: e.g. "facebook/whisper-large"
        work_dir: path to local directory where to store snapshot
        revision: optional commit / branch / tag
        force_download: if True, enforce re-download if remote snapshot differs
        **kwargs: other snapshot_download arguments (token, repo_type, allow_patterns, etc.)

    Returns:
        The path to the local snapshot folder (i.e. work_dir)
    """
    # First try local-only (no network)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=work_dir,
            local_files_only=True,
            **kwargs,
        )
        logging.info(f"Using existing local snapshot for {repo_id} at {path}")
        return path
    except LocalEntryNotFoundError:
        # Local snapshot doesn't exist — go ahead and download
        logging.info(f"No local snapshot found for {repo_id}. Downloading now...")
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=work_dir,
            force_download=force_download,
            **kwargs,
        )
        logging.info(f"Downloaded snapshot for {repo_id} to {path}")
        return path
