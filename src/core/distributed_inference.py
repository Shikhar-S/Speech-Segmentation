import os
import hydra
import torch
import logging
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import json


def work_chunk_(args, dataset, inference_config, inference_call_args=None, device=None):
    """Worker function to run inference on a chunk of data."""
    worker_id, idxs = args
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device != "cpu":
        device = f"cuda:{worker_id % torch.cuda.device_count()}"

    model = hydra.utils.instantiate(inference_config)
    out = []
    for i in tqdm(idxs, desc="Processing", leave=False):
        it = dataset[i]
        pred = model(
            **(inference_call_args or {}), **it
        )  # keys from dataset override those in inference_call_args
        out.append((i, pred))
    return out


def save_json(data, out_file):
    """Save data to a json file"""
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved: {out_file}")


def run_distributed_inference_(
    dataset,
    inference_config,
    inference_call_args=None,
    num_workers: int = 1,
    out_file=None,
):
    """Splits dataset and runs inference in parallel workers."""

    # fail fast
    assert out_file, "Please provide an out_file to save results."
    if os.path.exists(out_file):
        logging.error(f"Output file {out_file} already exists.")
    else:
        open(out_file, "w").close()

    device = inference_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(
        f"Running inference on {len(dataset)} utterances on"
        f" {device} with {num_workers} workers."
    )
    # split items from dataset, run against inference object replicas, gather results

    N = len(dataset)
    if num_workers <= 1:
        return work_chunk_(
            args=(0, range(N)),
            dataset=dataset,
            inference_config=inference_config,
            inference_call_args=inference_call_args,
            device=device,
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
        device=device,
    )

    with mp.get_context("spawn").Pool(num_workers) as pool:
        parts = list(
            tqdm(pool.imap(worker, enumerate(chunks)), total=len(chunks), desc="Chunks")
        )
    out = []
    for p in parts:
        out.extend(p)
    logging.info("Finished distributed inference.")
    out.sort(key=lambda x: x[0])
    save_json({x[0]: x[1] for x in out}, out_file)
    return out
