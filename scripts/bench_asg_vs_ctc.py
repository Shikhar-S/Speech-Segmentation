"""Benchmark ASG (DP and k2) vs CTC loss: forward + backward.

Usage:
    python scripts/bench_asg_vs_ctc.py
    python scripts/bench_asg_vs_ctc.py --device cuda
"""

import argparse
import time

import torch
import torch.nn as nn

from src.recipe.segment_recognize.losses.asg import (
    AutoSegmentationCriterion,
    AutoSegmentationCriterionK2,
    AutoSegmentationCriterionTriton,
)


def bench(fn, warmup=3, repeats=10):
    """Time a function, return (mean, std) in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def make_data(B, T, L, V, device):
    """Create matching inputs for both ASG and CTC."""
    emissions = torch.randn(
        B, T, V, device=device, requires_grad=True,
    )
    targets = torch.randint(1, V, (B, L), device=device)
    hlens = torch.full(
        (B,), T, dtype=torch.long, device=device,
    )
    ylens = torch.full(
        (B,), L, dtype=torch.long, device=device,
    )
    return emissions, targets, hlens, ylens


def run_asg(model, emissions, targets, hlens, ylens):
    """Forward + backward for ASG."""
    if emissions.grad is not None:
        emissions.grad = None
    if model.transitions.grad is not None:
        model.transitions.grad = None
    loss = model(emissions, targets, hlens, ylens)
    loss.sum().backward()


def run_ctc(ctc_lo, emissions, targets, hlens, ylens):
    """Forward + backward for CTC."""
    if emissions.grad is not None:
        emissions.grad = None
    ctc_lo.zero_grad()
    logits = ctc_lo(emissions)
    log_p = logits.log_softmax(dim=-1).permute(1, 0, 2)
    targets_clean = targets.clamp(min=0)
    loss = nn.functional.ctc_loss(
        log_p, targets_clean, hlens, ylens,
        blank=0, reduction="mean",
    )
    loss.backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    try:
        import k2  # noqa: F401
        has_k2 = True
    except ImportError:
        has_k2 = False

    try:
        import triton  # noqa: F401
        has_triton = device == "cuda"
    except ImportError:
        has_triton = False

    configs = [
        # (B, T, L, V, label)
        (1,  50,   5,  10, "B=1  T=50  L=5  V=10"),
        (1,  200,  10, 50, "B=1  T=200 L=10 V=50"),
        (4,  100,  8,  30, "B=4  T=100 L=8  V=30"),
        (8,  100,  10, 50, "B=8  T=100 L=10 V=50"),
        (16, 50,   5,  50, "B=16 T=50  L=5  V=50"),
        (1,  100,  5, 100, "B=1  T=100 L=5  V=100"),
    ]

    # Build header dynamically
    cols = f"{'Config':<28} {'ASG-DP (ms)':>20} "
    if has_triton:
        cols += f"{'ASG-Triton (ms)':>20} "
    if has_k2:
        cols += f"{'ASG-k2 (ms)':>20} "
    cols += f"{'CTC (ms)':>20} {'DP/CTC':>10}"
    if has_triton:
        cols += f"{'Tri/CTC':>10}"
    if has_k2:
        cols += f"{'k2/CTC':>10}"

    print(f"\nDevice: {device}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print("=" * len(cols))
    print(cols)
    print("-" * len(cols))

    for B, T, L, V, label in configs:
        emissions, targets, hlens, ylens = make_data(
            B, T, L, V, device,
        )

        # ASG (DP)
        asg_dp = AutoSegmentationCriterion(
            num_labels=V,
            use_transitions=True,
            use_double_scores=False,
        ).to(device)
        dp_mean, dp_std = bench(
            lambda: run_asg(
                asg_dp, emissions, targets, hlens, ylens,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        # ASG (Triton)
        tri_mean, tri_std = None, None
        if has_triton:
            asg_tri = AutoSegmentationCriterionTriton(
                num_labels=V,
                use_transitions=True,
                use_double_scores=False,
            ).to(device)
            tri_mean, tri_std = bench(
                lambda: run_asg(
                    asg_tri, emissions, targets,
                    hlens, ylens,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )

        # ASG (k2)
        k2_mean, k2_std = None, None
        if has_k2:
            asg_k2 = AutoSegmentationCriterionK2(
                num_labels=V,
                use_transitions=True,
                use_double_scores=False,
            ).to(device)
            k2_mean, k2_std = bench(
                lambda: run_asg(
                    asg_k2, emissions, targets,
                    hlens, ylens,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )

        # CTC
        ctc_lo = nn.Linear(V, V).to(device)
        ctc_mean, ctc_std = bench(
            lambda: run_ctc(
                ctc_lo, emissions, targets, hlens, ylens,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        row = (
            f"{label:<28} "
            f"{dp_mean:8.2f} +/- {dp_std:5.2f}   "
        )
        if has_triton:
            row += (
                f"{tri_mean:8.2f} +/- {tri_std:5.2f}   "
            )
        if has_k2:
            row += (
                f"{k2_mean:8.2f} +/- {k2_std:5.2f}   "
            )
        row += f"{ctc_mean:8.2f} +/- {ctc_std:5.2f}   "
        dp_ratio = dp_mean / max(ctc_mean, 1e-6)
        row += f"{dp_ratio:8.1f}x"
        if has_triton:
            tri_ratio = tri_mean / max(ctc_mean, 1e-6)
            row += f"{tri_ratio:8.1f}x"
        if has_k2:
            k2_ratio = k2_mean / max(ctc_mean, 1e-6)
            row += f"{k2_ratio:8.1f}x"
        print(row)

    print("=" * len(cols))
    notes = "\nASG-DP: batched PyTorch DP."
    if has_triton:
        notes += " ASG-Triton: fused Triton kernel."
    if has_k2:
        notes += " ASG-k2: k2 FSA intersection."
    notes += " CTC: PyTorch fused C++ kernel.\n"
    print(notes)


if __name__ == "__main__":
    main()
