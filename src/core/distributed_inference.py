from dataclasses import fields, is_dataclass
import os
import hydra
import torch
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import json

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
import logging as pylogging
from lightning_utilities.core.rank_zero import rank_zero_only


def _init_worker():
    proc = mp.current_process()
    rank_zero_only.rank = (proc._identity[0] - 1) if proc._identity else 0
    pylogging.basicConfig(
        level=pylogging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    )


def work_chunk_(
    args,
    dataset,
    inference_config,
    inference_call_args=None,
    passthrough_keys=None,
    device=None,
):
    """Worker function to run inference on a chunk of data."""
    log = RankedLogger(f"Worker-{args[0]}", rank_zero_only=False)

    worker_id, idxs = args
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device != "cpu":
        device = f"cuda:{worker_id % torch.cuda.device_count()}"

    log.info(f"Worker {worker_id} processing {len(idxs)} items on device {device}.")
    inference_obj = hydra.utils.instantiate(inference_config, device=device)
    out = []
    for i in tqdm(idxs, desc="Processing", leave=False):
        it = dataset[i]
        # keys from dataset override those in inference_call_args
        call_args = {**(inference_call_args or {}), **it}
        pred = inference_obj(**call_args)
        # keys from dataset that must be passthroughly passed to
        # output to be written
        out.append((i, pred, {k: it[k] for k in (passthrough_keys or []) if k in it}))
    return out


def default_encoder(o):
    if is_dataclass(o):
        return {f.name: getattr(o, f.name) for f in fields(o)}
    return str(o)


def save_json(data, out_file):
    """Save data to a json file"""
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default_encoder)
    log.info(f"Saved: {out_file}")


def load_json(in_file):
    """Load data from a json file"""
    with open(in_file, "r") as f:
        data = json.load(f)
    log.info(f"Loaded: {in_file}")
    return data


def run_distributed_inference_(
    dataset,
    inference_config,
    inference_call_args=None,
    num_workers: int = 1,
    out_file=None,
    passthrough_keys=[],
    limit_samples: int = None,
):
    """Splits dataset and runs inference in parallel workers.

    Args:
        dataset: Dataset object with __len__ and __getitem__
        inference_config: config for inference object to be instantiated in each worker
        inference_call_args: additional args to be passed to inference __call__ method
        num_workers: number of parallel workers
        out_file: output file to save results
        passthrough_keys: list of keys in dataset item to be written directly to
            output without processing
        limit_samples: if set, limit the number of samples to process (useful for testing)
    """

    # fail fast
    assert out_file, "Please provide an out_file to save results."
    if os.path.exists(out_file):
        log.error(f"Output file {out_file} already exists.")
    else:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        open(out_file, "w").close()

    device = inference_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate effective number of samples
    N = len(dataset)
    if limit_samples is not None and limit_samples > 0:
        N = min(N, limit_samples)
        log.info(f"Limiting inference to {N} samples (out of {len(dataset)} total).")

    log.info(
        f"Running inference on {N} utterances on"
        f" {device} with {num_workers} workers."
    )
    cs = (N + num_workers - 1) // num_workers
    chunks = [
        range(i * cs, min((i + 1) * cs, N)) for i in range(num_workers) if i * cs < N
    ]
    worker = partial(
        work_chunk_,
        dataset=dataset,
        inference_config=inference_config,
        inference_call_args=inference_call_args,
        passthrough_keys=passthrough_keys,
        device=device,
    )

    # TODO(shikhar): switch to jsonl append pattern
    with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker) as pool:
        out = []
        worker_id = 0
        for p in tqdm(
            pool.imap_unordered(worker, enumerate(chunks)), total=len(chunks)
        ):
            out.extend(p)

            # collect incrementally
            save_json(
                {
                    i: {"pred": pred, "passthrough": passthrough}
                    for i, pred, passthrough in p
                },
                f"{out_file}.part{worker_id}.json",
            )
            worker_id += 1
    log.info("Finished distributed inference.")

    # collect all results
    merged = {}
    for w_id in range(num_workers):
        part = load_json(f"{out_file}.part{w_id}.json")
        merged.update(part)
    save_json(merged, out_file)
    log.info(f"Saved final output to {out_file}.")

    # cleanup partial files
    log.info("Cleaning up partial files.")
    for w_id in range(num_workers):
        part_file = f"{out_file}.part{w_id}.json"
        if os.path.exists(part_file):
            os.remove(part_file)

    return out
