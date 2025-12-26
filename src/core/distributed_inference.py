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

    with open(out_file, "w") as f:
        f.write("{\n")
        is_first_record = True
        with mp.get_context("spawn").Pool(
            num_workers, initializer=_init_worker
        ) as pool:
            for chunk_results in tqdm(
                pool.imap_unordered(worker, enumerate(chunks)),
                total=len(chunks),
                desc="Workers Progress",
            ):
                for i, pred, passthrough in chunk_results:
                    # Write a comma before every record
                    # except the very first one
                    if not is_first_record:
                        f.write(",\n")
                    # Prepare the data for this specific key
                    record_payload = {"pred": pred, "passthrough": passthrough}
                    # Serialize only this specific record to a string
                    json_payload = json.dumps(
                        record_payload,
                        default=default_encoder,
                        indent=2,
                        ensure_ascii=False,
                    )
                    indented_payload = json_payload.replace("\n", "\n  ")
                    f.write(f'  "{i}": {indented_payload}')
                    is_first_record = False
        f.write("\n}")
    log.info(f"Finished distributed inference. Final output saved to {out_file}.")
