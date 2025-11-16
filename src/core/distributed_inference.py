import os
import hydra
import torch
import logging
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import json


def work_chunk_(
    args,
    dataset,
    inference_config,
    inference_call_args=None,
    passthrough_keys=None,
    device=None,
):
    """Worker function to run inference on a chunk of data."""
    worker_id, idxs = args
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device != "cpu":
        device = f"cuda:{worker_id % torch.cuda.device_count()}"

    model = hydra.utils.instantiate(inference_config)
    out = []
    for i in tqdm(idxs, desc="Processing", leave=False):
        it = dataset[i]
        # keys from dataset override those in inference_call_args
        pred = model(**(inference_call_args or {}), **it)
        # keys from dataset that must be passthroughly passed to
        # output to be written
        out.append((i, pred, {k: it[k] for k in (passthrough_keys or []) if k in it}))
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
    passthrough_keys=[],
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
    """

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
    N = 2
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

    with mp.get_context("spawn").Pool(num_workers) as pool:
        parts = list(
            tqdm(pool.imap(worker, enumerate(chunks)), total=len(chunks), desc="Chunks")
        )
    out = []
    for p in parts:
        out.extend(p)
    logging.info("Finished distributed inference.")
    save_json(
        {i: {"pred": pred, "passthrough": passthrough} for i, pred, passthrough in out},
        out_file,
    )
    return out
