"""Triton-fused ASG forward and backward kernels.

Each kernel fuses the entire T-step DP into a single GPU kernel
launch, eliminating per-step kernel launch overhead.  One thread
block handles one batch element; the DP state (beta/alpha vectors)
lives in registers across timesteps.

.. warning::

    **Not optimized for large V.**  The denominator kernels use a
    ``(BLOCK_V, BLOCK_V)`` 2-D tile that must fit in registers.
    For V > ~128 the tile spills to local memory and performance
    degrades below the batched-PyTorch DP baseline.  At our
    production vocab size (V = 470) the batched DP in
    :class:`AutoSegmentationCriterion` is faster.

Benchmark results (GH200, fwd+bwd, ms) — April 2026:

Small V (V <= 100) — Triton wins
::

    Config                  DP    Triton   k2     CTC
    B=1  T=50  L=5  V=10   39.0   1.3     9.0    0.8
    B=1  T=200 L=10 V=50  157.2   3.4    17.0    0.8
    B=4  T=100 L=8  V=30   76.8   1.8    14.2    0.8
    B=8  T=100 L=10 V=50   88.6   2.5    18.2    0.7
    B=16 T=50  L=5  V=50   40.5   1.9    21.9    0.7
    B=1  T=100 L=5  V=100  78.3   3.0    15.8    0.7

Large V (V = 470) — batched DP wins
::

    Config                  DP    Triton   k2     CTC
    B=1  T=100 L=10        83.5   95.1   288.8    0.7
    B=1  T=200 L=20       160.8  190.9   208.5    0.9
    B=4  T=100 L=10        83.4   96.5   256.7    0.9
    B=8  T=100 L=10        87.3  100.7   342.2    1.2
    B=16 T=50  L=10        42.8   57.7   256.4    1.3
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------
# Triton kernels
# -------------------------------------------------------------------


@triton.jit
def _den_fwd_kernel(
    em_ptr,        # (B, T, V)
    trans_ptr,     # (V, V)
    hlens_ptr,     # (B,)
    beta_ptr,      # (B, T, V) — saved for backward
    den_ptr,       # (B,)
    T: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Denominator forward using 2-D tiles for the V×V step."""
    b = tl.program_id(0)
    hlen = tl.load(hlens_ptr + b)
    vi = tl.arange(0, BLOCK_V)            # source
    vj = tl.arange(0, BLOCK_V)            # dest
    mask_j = vj < V
    NEG = -1e10

    # Load transition matrix once: (BLOCK_V, BLOCK_V)
    offs_2d = vi[:, None] * V + vj[None, :]
    mask_2d = (vi[:, None] < V) & (vj[None, :] < V)
    trans = tl.load(
        trans_ptr + offs_2d, mask=mask_2d, other=NEG,
    )

    beta = tl.load(
        em_ptr + b * T * V + vj, mask=mask_j, other=NEG,
    )
    tl.store(
        beta_ptr + b * T * V + vj, beta, mask=mask_j,
    )

    for t in range(1, T):
        em_t = tl.load(
            em_ptr + b * T * V + t * V + vj,
            mask=mask_j, other=NEG,
        )
        # 2-D: val[i,j] = beta[i] + trans[i,j]
        val = tl.expand_dims(beta, 1) + trans  # (V, V)
        # logsumexp over i (axis 0) → (V,)
        mx = tl.max(tl.where(mask_2d, val, NEG), axis=0)
        lse = mx + tl.log(
            tl.sum(
                tl.where(
                    mask_2d,
                    tl.exp(val - tl.expand_dims(mx, 0)),
                    0.0,
                ),
                axis=0,
            ),
        )
        beta_new = lse + em_t
        beta = tl.where(t < hlen, beta_new, beta)
        tl.store(
            beta_ptr + b * T * V + t * V + vj,
            beta, mask=mask_j,
        )

    # Final logsumexp.
    mx = tl.max(tl.where(mask_j, beta, NEG), axis=0)
    s = tl.sum(
        tl.where(mask_j, tl.exp(beta - mx), 0.0), axis=0,
    )
    tl.store(den_ptr + b, mx + tl.log(s))


@triton.jit
def _num_fwd_kernel(
    em_ptr,        # (B, T, V)
    tgt_ptr,       # (B, L)
    hlens_ptr,     # (B,)
    ylens_ptr,     # (B,)
    trans_ptr,     # (V, V)
    alpha_ptr,     # (B, T, L) — saved for backward
    num_ptr,       # (B,)
    T: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Numerator forward with vectorized gathers."""
    b = tl.program_id(0)
    hlen = tl.load(hlens_ptr + b)
    ylen = tl.load(ylens_ptr + b)
    idx = tl.arange(0, BLOCK_L)
    mask = idx < ylen
    NEG = -1e10

    # Load target labels — vectorized
    tgt = tl.load(
        tgt_ptr + b * L + idx, mask=mask, other=0,
    )
    tgt_prev = tl.load(
        tgt_ptr + b * L + tl.maximum(idx - 1, 0),
        mask=mask, other=0,
    )

    # Vectorized gather for transition scores
    stay_tr = tl.load(
        trans_ptr + tgt * V + tgt,
        mask=mask, other=0.0,
    ).to(tl.float32)
    adv_tr = tl.load(
        trans_ptr + tgt_prev * V + tgt,
        mask=mask & (idx > 0), other=0.0,
    ).to(tl.float32)

    # Gather emission at t=0
    emit_0 = tl.load(
        em_ptr + b * T * V + tgt,
        mask=mask, other=NEG,
    ).to(tl.float32)
    alpha = tl.where(idx == 0, emit_0, NEG)
    tl.store(
        alpha_ptr + b * T * L + idx, alpha, mask=mask,
    )

    for t in range(1, T):
        # Vectorized emission gather
        emit = tl.load(
            em_ptr + b * T * V + t * V + tgt,
            mask=mask, other=NEG,
        ).to(tl.float32)

        stay = alpha + stay_tr
        # Shift alpha right by 1 for advance
        adv = tl.full([BLOCK_L], NEG, dtype=tl.float32)
        for i in range(1, BLOCK_L):
            if i < ylen:
                a_prev = tl.sum(
                    tl.where(idx == i - 1, alpha, 0.0),
                )
                adv = tl.where(
                    idx == i, a_prev + adv_tr, adv,
                )
        alpha_next = tl.where(
            mask,
            _logaddexp(stay, adv) + emit,
            NEG,
        )
        alpha = tl.where(t < hlen, alpha_next, alpha)
        tl.store(
            alpha_ptr + b * T * L + t * L + idx,
            alpha, mask=mask,
        )

    result = tl.sum(
        tl.where(idx == ylen - 1, alpha, 0.0),
    )
    tl.store(num_ptr + b, result)


@triton.jit
def _den_bwd_kernel(
    em_ptr,        # (B, T, V)
    trans_ptr,     # (V, V)
    hlens_ptr,     # (B,)
    beta_ptr,      # (B, T, V) — from forward
    d_den_ptr,     # (B,) — incoming gradient
    d_em_ptr,      # (B, T, V) — output grad
    d_trans_ptr,   # (V, V) — shared, uses atomic_add
    T: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Denominator backward using 2-D tiles."""
    b = tl.program_id(0)
    hlen = tl.load(hlens_ptr + b)
    vi = tl.arange(0, BLOCK_V)
    vj = tl.arange(0, BLOCK_V)
    mask_i = vi < V
    mask_j = vj < V
    mask_2d = (vi[:, None] < V) & (vj[None, :] < V)
    NEG = -1e10

    d_loss = tl.load(d_den_ptr + b)

    offs_2d = vi[:, None] * V + vj[None, :]
    trans = tl.load(
        trans_ptr + offs_2d, mask=mask_2d, other=NEG,
    )

    base = beta_ptr + b * T * V
    em_base = em_ptr + b * T * V

    beta_last = tl.load(
        base + (hlen - 1) * V + vj,
        mask=mask_j, other=NEG,
    )
    mx = tl.max(
        tl.where(mask_j, beta_last, NEG), axis=0,
    )
    Z = tl.sum(
        tl.where(mask_j, tl.exp(beta_last - mx), 0.0),
        axis=0,
    )
    d_beta = tl.where(
        mask_j, d_loss * tl.exp(beta_last - mx) / Z, 0.0,
    )

    for t_off in range(T):
        t = T - 1 - t_off
        if t < hlen:
            tl.store(
                d_em_ptr + b * T * V + t * V + vj,
                d_beta, mask=mask_j,
            )
            if t > 0:
                beta_prev = tl.load(
                    base + (t - 1) * V + vi,
                    mask=mask_i, other=NEG,
                )
                em_t = tl.load(
                    em_base + t * V + vj,
                    mask=mask_j, other=NEG,
                )
                beta_cur = tl.load(
                    base + t * V + vj,
                    mask=mask_j, other=NEG,
                )
                log_Z = beta_cur - em_t

                w = tl.where(
                    mask_2d,
                    tl.exp(
                        tl.expand_dims(beta_prev, 1)
                        + trans
                        - tl.expand_dims(log_Z, 0)
                    ),
                    0.0,
                )

                grad = w * tl.expand_dims(d_beta, 0)

                d_beta = tl.sum(grad, axis=1)
                d_beta = tl.where(mask_i, d_beta, 0.0)

                # Atomic accumulate to shared d_trans
                tl.atomic_add(
                    d_trans_ptr + offs_2d,
                    tl.where(mask_2d, grad, 0.0),
                )


@triton.jit
def _num_bwd_kernel(
    em_ptr,        # (B, T, V)
    tgt_ptr,       # (B, L)
    hlens_ptr,     # (B,)
    ylens_ptr,     # (B,)
    trans_ptr,     # (V, V)
    alpha_ptr,     # (B, T, L) — from forward
    d_num_ptr,     # (B,) — incoming gradient (negated)
    d_em_ptr,      # (B, T, V) — output grad (accumulate)
    d_trans_ptr,   # (V, V) — shared, uses atomic_add
    T: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Numerator backward with vectorized gathers."""
    b = tl.program_id(0)
    hlen = tl.load(hlens_ptr + b)
    ylen = tl.load(ylens_ptr + b)
    idx = tl.arange(0, BLOCK_L)
    mask = idx < ylen
    NEG = -1e10

    d_loss = tl.load(d_num_ptr + b)

    tgt = tl.load(
        tgt_ptr + b * L + idx, mask=mask, other=0,
    )
    tgt_prev = tl.load(
        tgt_ptr + b * L + tl.maximum(idx - 1, 0),
        mask=mask, other=0,
    )

    # Vectorized transition gather
    stay_tr = tl.load(
        trans_ptr + tgt * V + tgt,
        mask=mask, other=0.0,
    ).to(tl.float32)
    adv_tr = tl.load(
        trans_ptr + tgt_prev * V + tgt,
        mask=mask & (idx > 0), other=0.0,
    ).to(tl.float32)

    d_alpha = tl.where(idx == ylen - 1, d_loss, 0.0)
    a_base = alpha_ptr + b * T * L

    for t_off in range(T):
        t = T - 1 - t_off
        if t < hlen:
            # Scatter d_alpha to emission grads
            tl.atomic_add(
                d_em_ptr + b * T * V + t * V + tgt,
                tl.where(mask, d_alpha, 0.0),
            )

            if t > 0:
                alpha_prev = tl.load(
                    a_base + (t - 1) * L + idx,
                    mask=mask, other=NEG,
                )
                alpha_cur = tl.load(
                    a_base + t * L + idx,
                    mask=mask, other=NEG,
                )
                # Vectorized emission gather
                emit = tl.load(
                    em_ptr + b * T * V + t * V + tgt,
                    mask=mask, other=0.0,
                ).to(tl.float32)

                log_z = alpha_cur - emit

                s_val = alpha_prev + stay_tr
                # Advance shift (still needs scalar loop)
                a_val = tl.full(
                    [BLOCK_L], NEG, dtype=tl.float32,
                )
                for i in range(1, BLOCK_L):
                    if i < ylen:
                        ap = tl.sum(tl.where(
                            idx == i - 1, alpha_prev, 0.0,
                        ))
                        a_val = tl.where(
                            idx == i, ap + adv_tr, a_val,
                        )

                w_stay = tl.where(
                    mask, tl.exp(s_val - log_z), 0.0,
                )
                w_adv = tl.where(
                    mask, tl.exp(a_val - log_z), 0.0,
                )

                d_alpha_prev = d_alpha * w_stay
                for i in range(1, BLOCK_L):
                    if i < ylen:
                        c = tl.sum(tl.where(
                            idx == i,
                            d_alpha * w_adv, 0.0,
                        ))
                        d_alpha_prev = tl.where(
                            idx == i - 1,
                            d_alpha_prev + c,
                            d_alpha_prev,
                        )

                # Transition grads — vectorized scatter
                d_stay = d_alpha * w_stay
                d_adv = d_alpha * w_adv
                tl.atomic_add(
                    d_trans_ptr + tgt * V + tgt,
                    tl.where(mask, d_stay, 0.0),
                )
                tl.atomic_add(
                    d_trans_ptr + tgt_prev * V + tgt,
                    tl.where(
                        mask & (idx > 0), d_adv, 0.0,
                    ),
                )

                d_alpha = d_alpha_prev


@triton.jit
def _logaddexp(a, b):
    mx = tl.maximum(a, b)
    return mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))


# -------------------------------------------------------------------
# Autograd wrapper
# -------------------------------------------------------------------


class _ASGLoss(torch.autograd.Function):
    """Fused ASG forward + backward via Triton."""

    @staticmethod
    def forward(
        ctx,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        hlens: torch.Tensor,
        ylens: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = emissions.shape
        L = targets.shape[1]

        BLOCK_V = triton.next_power_of_2(V)
        BLOCK_L = triton.next_power_of_2(L)

        em = emissions.contiguous().float()
        trans = transitions.contiguous().float()
        tgt = targets.contiguous()

        # Allocate outputs and buffers
        den = torch.empty(B, device=em.device, dtype=em.dtype)
        num = torch.empty(B, device=em.device, dtype=em.dtype)
        beta_buf = torch.empty(
            B, T, V, device=em.device, dtype=em.dtype,
        )
        alpha_buf = torch.empty(
            B, T, L, device=em.device, dtype=em.dtype,
        )

        _den_fwd_kernel[(B,)](
            em, trans, hlens, beta_buf, den,
            T=T, V=V, BLOCK_V=BLOCK_V,
        )
        _num_fwd_kernel[(B,)](
            em, tgt, hlens, ylens, trans,
            alpha_buf, num,
            T=T, L=L, V=V, BLOCK_L=BLOCK_L,
        )

        ctx.save_for_backward(
            em, trans, tgt, hlens, ylens,
            beta_buf, alpha_buf,
        )
        ctx.T = T
        ctx.V = V
        ctx.L = L
        loss = -num + den
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, None, None, None, torch.Tensor,
    ]:
        (
            em, trans, tgt, hlens, ylens,
            beta_buf, alpha_buf,
        ) = ctx.saved_tensors
        B = em.shape[0]
        T, V, L = ctx.T, ctx.V, ctx.L
        BLOCK_V = triton.next_power_of_2(V)

        d_em = torch.zeros_like(em)
        d_trans = torch.zeros(
            V, V, device=em.device, dtype=em.dtype,
        )
        neg_grad = -grad_output

        _den_bwd_kernel[(B,)](
            em, trans, hlens, beta_buf,
            grad_output,
            d_em, d_trans,
            T=T, V=V, BLOCK_V=BLOCK_V,
        )

        BLOCK_L = triton.next_power_of_2(L)
        _num_bwd_kernel[(B,)](
            em, tgt, hlens, ylens, trans,
            alpha_buf, neg_grad,
            d_em, d_trans,
            T=T, L=L, V=V, BLOCK_L=BLOCK_L,
        )

        return d_em, None, None, None, d_trans


def _numerator_pytorch(
    em: torch.Tensor,
    tgt: torch.Tensor,
    hlens: torch.Tensor,
    ylens: torch.Tensor,
    trans: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Batched numerator in pure PyTorch (for backward)."""
    B, T, V = em.shape
    NEG = -1e10
    idx = tgt.unsqueeze(1).expand(-1, T, -1)
    emit = em.gather(2, idx)                      # (B, T, L)
    stay_tr = trans[tgt, tgt]                     # (B, L)
    prev_t = torch.cat(
        [tgt[:, :1], tgt[:, :-1]], dim=1,
    )
    adv_tr = trans[prev_t, tgt]                   # (B, L)

    alpha = torch.full(
        (B, L), NEG, device=em.device, dtype=em.dtype,
    )
    alpha[:, 0] = emit[:, 0, 0]

    for t in range(1, T):
        stay = alpha + stay_tr
        adv = torch.full_like(alpha, NEG)
        adv[:, 1:] = alpha[:, :-1] + adv_tr[:, 1:]
        alpha_next = (
            torch.logaddexp(stay, adv) + emit[:, t, :]
        )
        mask = (t < hlens).unsqueeze(1)
        alpha = torch.where(mask, alpha_next, alpha)

    return alpha[
        torch.arange(B, device=alpha.device),
        ylens - 1,
    ]


def _denominator_pytorch(
    em: torch.Tensor,
    hlens: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    """Batched denominator in pure PyTorch (for backward)."""
    B, T, _ = em.shape
    beta = em[:, 0, :]                              # (B, V)
    for t in range(1, T):
        beta_next = (
            torch.logsumexp(
                beta.unsqueeze(2) + trans, dim=1,
            )
            + em[:, t, :]
        )
        mask = (t < hlens).unsqueeze(1)
        beta = torch.where(mask, beta_next, beta)
    return torch.logsumexp(beta, dim=1)


def asg_loss(
    emissions: torch.Tensor,
    targets: torch.Tensor,
    hlens: torch.Tensor,
    ylens: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """Compute ASG loss via fused Triton kernels.

    Args:
        emissions: ``(B, T, V)`` emission scores.
        targets: ``(B, L)`` padded target label indices.
        hlens: ``(B,)`` emission lengths.
        ylens: ``(B,)`` target lengths.
        transitions: ``(V, V)`` transition scores.

    Returns:
        ``(B,)`` per-utterance loss.
    """
    return _ASGLoss.apply(
        emissions, targets, hlens, ylens, transitions,
    )
