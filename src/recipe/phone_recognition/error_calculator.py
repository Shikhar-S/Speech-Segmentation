# ~2-3x Faster error calculator!
import torch
import numpy as np
import time
from rapidfuzz.distance import Levenshtein
from typing import List
from espnet_import.nets.e2e_asr_common import ErrorCalculator as ESPnetErrorCalculator


class ErrorCalculator:
    def __init__(
        self,
        token_list: List[str],
        blank_id: int,
        sym_space: str = "<space>",
        ignore_id: int = -1,
    ):
        self.token_list = np.array(token_list)
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.idx_space = (
            token_list.index(sym_space) if sym_space in token_list else None
        )
        self.ignore_set = {ignore_id, blank_id, self.idx_space}

    def _ctc_collapse_batch(self, x: torch.Tensor):
        if x.numel() == 0:
            return x, torch.zeros(0, device=x.device, dtype=torch.long)

        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 1:] = x[:, 1:] != x[:, :-1]

        lengths = mask.sum(1)
        max_len = int(lengths.max().item())
        out = torch.full(
            (x.size(0), max_len), self.ignore_id, device=x.device, dtype=x.dtype
        )

        pos = mask.long().cumsum(1) - 1
        batch_offsets = torch.arange(x.size(0), device=x.device).unsqueeze(1) * max_len
        flat_dest_indices = (pos + batch_offsets)[mask]
        out.view(-1)[flat_dest_indices] = x[mask]

        return out, lengths

    def _ids_to_strs(self, ids_cpu: np.ndarray, lens_cpu: np.ndarray) -> List[str]:
        """Lookup-Table optimized string builder."""
        results = []
        # Optimization: Use a local variable for the array
        t_list = self.token_list
        i_id, b_id, s_id = self.ignore_id, self.blank_id, self.idx_space

        for i in range(len(ids_cpu)):
            row = ids_cpu[i, : lens_cpu[i]]
            # Vectorized filtering via Numpy is much faster than Python 'if' for large rows
            filtered = row[(row != i_id) & (row != b_id) & (row != s_id)]
            results.append("".join(t_list[filtered]))
        return results

    def __call__(
        self,
        ys_hat: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        verbose: bool = True,
    ):
        t = {}

        # 1. GPU Collapse
        t0 = time.perf_counter()
        collapsed_hat, hat_lens = self._ctc_collapse_batch(ys_hat)
        if ys_hat.is_cuda:
            torch.cuda.synchronize()
        t["collapse"] = (time.perf_counter() - t0) * 1000

        # 2. Transfer
        t0 = time.perf_counter()
        hat_np = collapsed_hat.cpu().numpy()
        hat_lens_np = hat_lens.cpu().numpy()
        ref_np = ys_pad.cpu().numpy()
        ref_lens_np = ys_pad_lens.cpu().numpy()
        t["transfer"] = (time.perf_counter() - t0) * 1000

        # 3. Stringify
        t0 = time.perf_counter()
        hyps = self._ids_to_strs(hat_np, hat_lens_np)
        refs = self._ids_to_strs(ref_np, ref_lens_np)
        t["stringify"] = (time.perf_counter() - t0) * 1000

        # 4. Edit Distance
        t0 = time.perf_counter()
        # Filter pairs where ref is not empty
        pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]

        if not pairs:
            cer = 0.0
        else:
            v_hyp, v_ref = zip(*pairs)
            distances = [Levenshtein.distance(h, r) for h, r in zip(v_hyp, v_ref)]

            total_dist = sum(distances)
            total_ref_len = sum(len(r) for r in v_ref)
            cer = total_dist / total_ref_len
        t["edit_dist"] = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"\n[TIMING] Batch: {ys_hat.size(0)}")
            print(f"  Collapse:  {t['collapse']:8.2f}ms")
            print(f"  Transfer:  {t['transfer']:8.2f}ms")
            print(f"  Stringify: {t['stringify']:8.2f}ms")
            print(f"  Edit Dist: {t['edit_dist']:8.2f}ms (RapidFuzz Parallel)")
            print(f"  TOTAL:     {sum(t.values()):8.2f}ms")

        return cer


# --- BENCHMARK SUITE ---


def run_comparison():
    token_list = [f"char_{i}" for i in range(100)]
    token_list[0], token_list[1] = "<blank>", "<space>"

    BATCH_SIZE = 16
    SEQ_LEN = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize
    ref_calc = ESPnetErrorCalculator(token_list, "<space>", "<blank>", report_cer=True)
    opt_calc = ErrorCalculator(token_list, blank_id=0)

    # Data generation
    ys_hat = torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)).to(device)
    ys_pad = torch.randint(0, 100, (BATCH_SIZE, 100)).to(device)
    ys_pad_lens = torch.randint(10, 100, (BATCH_SIZE,)).to(device)
    for i in range(BATCH_SIZE):
        ys_pad[i, ys_pad_lens[i] :] = -1

    print(f"Starting Benchmark on {device}...")

    # 1. Benchmark ESPnet (Reference)
    hat_np, pad_np = ys_hat.cpu().numpy(), ys_pad.cpu().numpy()
    start = time.perf_counter()
    ref_cer = ref_calc(hat_np, pad_np, is_ctc=True)
    espnet_total = (time.perf_counter() - start) * 1000

    # 2. Benchmark Custom (Warmup)
    _ = opt_calc(ys_hat, ys_pad, ys_pad_lens, verbose=False)

    # 3. Benchmark Custom (Actual)
    start = time.perf_counter()
    opt_cer = opt_calc(ys_hat, ys_pad, ys_pad_lens, verbose=True)
    custom_total = (time.perf_counter() - start) * 1000

    print("\n" + "=" * 40)
    print(f"ESPnet Total: {espnet_total:10.2f} ms")
    print(f"Custom Total: {custom_total:10.2f} ms")
    print(f"SPEEDUP:      {espnet_total / custom_total:10.2f}x")
    print(f"CER Match:    {np.isclose(ref_cer, opt_cer)}")
    print("=" * 40)


if __name__ == "__main__":
    run_comparison()
