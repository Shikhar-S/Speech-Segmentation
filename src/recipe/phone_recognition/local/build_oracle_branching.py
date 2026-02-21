#!/usr/bin/env python3
"""Align epitran and reference phone sequences using panphon-weighted edit
distance, then build an epitran → reference phone substitution mapping.

Kaldi-style input format (one utterance per line):
    <utt_id> <unsegmented IPA string>

Phones are NOT space-separated.  panphon's ipa_segs() is used to segment
the continuous IPA string into individual phones.

Alignment cost model
--------------------
* **Substitution**: panphon weighted feature edit distance between two phones,
  normalised to [0, 1].  Identical phones cost 0; maximally different → 1.
* **Insertion / Deletion**: fixed cost of 1.0 (always >= any substitution).

This encourages the aligner to prefer substituting similar phones over
inserting or deleting them.

Outputs
-------
1. A JSON mapping: {epitran_phone: [[ref_phone, ratio], ...]}
   where ratio = substitution_count / total_occurrences_of_epitran_phone.
2. Rich table on stdout.

Usage
-----    
    python /work/nvme/bbjs/sbharadwaj/powsm/xeuspr/src/recipe/phone_recognition/local/build_oracle_branching.py \
        --ref   /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_epadb/text.good \
            /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_speechoceannotth/text.good \
            /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_l2arctic_perceived/text.good \
            /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_gmuaccent/text.good \
            /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_buckeye/text.good \
        --hyp   exp/data/epitran_outputs/epadb.epitran \
            exp/data/epitran_outputs/buckeye.epitran \
            exp/data/epitran_outputs/speechoceannotth.epitran \
            /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_l2arctic/text.good \
            exp/data/epitran_outputs/gmuaccent.epitran \
        --output exp/data/epitran_outputs/oracle_mapping.json \
        --threshold 0.01
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import panphon
import panphon.distance

from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Kaldi text I/O  +  panphon segmentation
# ---------------------------------------------------------------------------

_FT = panphon.FeatureTable()


def segment_ipa(text: str) -> List[str]:
    """Segment a continuous IPA string into phones using panphon."""
    return _FT.ipa_segs(text)


def load_kaldi_ipa(filepath: Union[str, List[str]]) -> Dict[str, List[str]]:
    """Load a Kaldi-style file with unsegmented IPA and return segmented phones.

    Returns {utt_id: [phone1, phone2, ...]}
    """
    utterances = {}
    for fp in filepath if isinstance(filepath, list) else [filepath]:
        with open(fp, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    print(
                        f"WARNING: skipping line {lineno} (no content): {line}",
                        file=sys.stderr,
                    )
                    continue
                utt_id = parts[0]
                raw_ipa = parts[1].replace(" ", "")  # collapse any spaces
                phones = segment_ipa(raw_ipa)
                utterances[utt_id] = phones
    return utterances


# ---------------------------------------------------------------------------
# Panphon-weighted alignment
# ---------------------------------------------------------------------------


class PanphonAligner:
    """Weighted edit-distance aligner using panphon feature distances."""

    INDEL_COST = 1.0  # fixed; always >= max substitution cost

    def __init__(self):
        self.dst = panphon.distance.Distance()
        self._sub_cache: Dict[Tuple[str, str], float] = {}

    def _substitution_cost(self, a: str, b: str) -> float:
        """Panphon weighted feature edit distance, normalised to [0, 1]."""
        if a == b:
            return 0.0
        key = (a, b)
        if key not in self._sub_cache:
            cost = self.dst.weighted_feature_edit_distance(a, b)
            cost = min(max(cost, 0.0), 1.0)
            self._sub_cache[key] = cost
            self._sub_cache[(b, a)] = cost
        return self._sub_cache[key]

    def align(
        self, hyp: List[str], ref: List[str]
    ) -> List[Tuple[Optional[str], Optional[str], str]]:
        """Align two phone sequences using weighted edit distance.

        Returns [(hyp_phone|None, ref_phone|None, op), ...]
        op ∈ {'match', 'sub', 'ins', 'del'}
        """
        H, R = len(hyp), len(ref)

        # DP
        D = [[0.0] * (R + 1) for _ in range(H + 1)]
        for i in range(1, H + 1):
            D[i][0] = D[i - 1][0] + self.INDEL_COST
        for j in range(1, R + 1):
            D[0][j] = D[0][j - 1] + self.INDEL_COST

        for i in range(1, H + 1):
            for j in range(1, R + 1):
                sub_cost = self._substitution_cost(hyp[i - 1], ref[j - 1])
                D[i][j] = min(
                    D[i - 1][j] + self.INDEL_COST,
                    D[i][j - 1] + self.INDEL_COST,
                    D[i - 1][j - 1] + sub_cost,
                )

        # Backtrace
        alignment = []
        i, j = H, R
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub_cost = self._substitution_cost(hyp[i - 1], ref[j - 1])
                if abs(D[i][j] - (D[i - 1][j - 1] + sub_cost)) < 1e-9:
                    op = "match" if sub_cost == 0.0 else "sub"
                    alignment.append((hyp[i - 1], ref[j - 1], op))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and abs(D[i][j] - (D[i - 1][j] + self.INDEL_COST)) < 1e-9:
                alignment.append((hyp[i - 1], None, "ins"))
                i -= 1
            elif j > 0 and abs(D[i][j] - (D[i][j - 1] + self.INDEL_COST)) < 1e-9:
                alignment.append((None, ref[j - 1], "del"))
                j -= 1
            else:
                if i > 0:
                    alignment.append((hyp[i - 1], None, "ins"))
                    i -= 1
                else:
                    alignment.append((None, ref[j - 1], "del"))
                    j -= 1

        alignment.reverse()
        return alignment


# ---------------------------------------------------------------------------
# Build substitution mapping
# ---------------------------------------------------------------------------


def build_mapping(
    hyp_data: Dict[str, List[str]],
    ref_data: Dict[str, List[str]],
    aligner: PanphonAligner,
    threshold: float = 0.0,
) -> Tuple[
    Dict[str, List[Tuple[str, float]]],  # mapping
    Counter,  # ins_counts
    Counter,  # del_counts
]:
    """Align all utterances and build the epitran → ref mapping.

    Returns:
        mapping:    {epitran_phone: [(ref_phone, ratio), ...]}
                    sorted by ratio descending, filtered by threshold.
        ins_counts: Counter of hyp phones with no ref counterpart.
        del_counts: Counter of ref phones with no hyp counterpart.

    ratio = count_of_this_substitution / total_occurrences_of_epitran_phone
    (total occurrences = matches + substitutions + insertions for that phone)
    """
    common_utts = sorted(set(hyp_data.keys()) & set(ref_data.keys()))
    hyp_only = set(hyp_data.keys()) - set(ref_data.keys())
    ref_only = set(ref_data.keys()) - set(hyp_data.keys())

    if hyp_only:
        print(
            f"WARNING: {len(hyp_only)} utterances in hyp only — skipped.",
            file=sys.stderr,
        )
    if ref_only:
        print(
            f"WARNING: {len(ref_only)} utterances in ref only — skipped.",
            file=sys.stderr,
        )
    print(f"Aligning {len(common_utts)} common utterances …")

    # Accumulate counts
    sub_counts: Dict[str, Counter] = defaultdict(Counter)  # epi → {ref: N}
    epi_total: Counter = Counter()  # total per epi phone
    ins_counts: Counter = Counter()
    del_counts: Counter = Counter()

    for utt_id in tqdm(common_utts, desc="Aligning"):
        alignment = aligner.align(hyp_data[utt_id], ref_data[utt_id])
        for hyp_ph, ref_ph, op in alignment:
            if op in ("match", "sub"):
                sub_counts[hyp_ph][ref_ph] += 1
                epi_total[hyp_ph] += 1
            elif op == "ins":
                ins_counts[hyp_ph] += 1
                epi_total[hyp_ph] += 1
            elif op == "del":
                del_counts[ref_ph] += 1

    # Build mapping with ratios
    mapping: Dict[str, List[Tuple[str, float]]] = {}
    for epi_ph in sorted(sub_counts.keys()):
        total = epi_total[epi_ph]
        pairs = [
            (ref_ph, count / total)
            for ref_ph, count in sub_counts[epi_ph].most_common()
        ]
        # Apply threshold
        if threshold > 0:
            pairs = [(ref_ph, ratio) for ref_ph, ratio in pairs if ratio >= threshold]
        if pairs:
            mapping[epi_ph] = pairs

    return mapping, ins_counts, del_counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def pretty_print_mapping(
    mapping: Dict[str, List[Tuple[str, float]]],
    top_n: int = 5,
) -> None:
    console = Console()

    t = Table(title="Epitran → Reference Phone Mapping")
    t.add_column("Epitran", style="bold cyan")
    t.add_column("→ Best Ref", style="bold green")
    t.add_column("Best Ratio", justify="right")
    t.add_column(f"Top-{top_n} Mappings (phone: ratio)")

    n_identity = 0
    for epi_ph, pairs in sorted(mapping.items(), key=lambda x: x[0]):
        best_ref, best_ratio = pairs[0]
        is_identity = epi_ph == best_ref
        if is_identity:
            n_identity += 1
        best_str = best_ref + (" ✓" if is_identity else "")
        alts = ", ".join(f"{ph}:{ratio:.2%}" for ph, ratio in pairs[:top_n])
        t.add_row(epi_ph, best_str, f"{best_ratio:.2%}", alts)

    console.print(t)
    n_total = len(mapping)
    console.print(
        f"\n[bold]Summary:[/bold] {n_total} epitran phones mapped, "
        f"{n_identity} map to themselves ({n_identity / n_total:.0%})"
    )


def pretty_print_indels(ins_counts: Counter, del_counts: Counter, top_n: int = 20):
    console = Console()
    if ins_counts:
        t = Table(title="Top Insertions (in epitran, not in ref)")
        t.add_column("Phone", style="bold red")
        t.add_column("Count", justify="right")
        for ph, c in ins_counts.most_common(top_n):
            t.add_row(ph, str(c))
        console.print(t)

    if del_counts:
        t = Table(title="Top Deletions (in ref, not in epitran)")
        t.add_column("Phone", style="bold yellow")
        t.add_column("Count", justify="right")
        for ph, c in del_counts.most_common(top_n):
            t.add_row(ph, str(c))
        console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Align epitran ↔ reference phone sequences (unsegmented IPA) "
        "using panphon-weighted edit distance and build a substitution "
        "mapping."
    )
    parser.add_argument(
        "--ref",
        nargs="+",
        required=True,
        help="Kaldi-style reference file (utt_id <IPA string>)",
    )
    parser.add_argument(
        "--hyp", nargs="+", required=True, help="Kaldi-style epitran/hypothesis file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON mapping file (default: epitran2ref_map.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min ratio to include a ref phone in the mapping "
        "(default: 0.0, keep all)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of alternatives to show in table (default: 5)",
    )
    args = parser.parse_args()

    # Load & segment
    print("Loading and segmenting reference …")
    ref_data = load_kaldi_ipa(args.ref)
    print(f"  {len(ref_data)} utterances")

    print("Loading and segmenting hypothesis (epitran) …")
    hyp_data = load_kaldi_ipa(args.hyp)
    print(f"  {len(hyp_data)} utterances")

    # Align & map
    aligner = PanphonAligner()
    mapping, ins_counts, del_counts = build_mapping(
        hyp_data, ref_data, aligner, threshold=args.threshold
    )

    # Display
    pretty_print_mapping(mapping, top_n=args.top_n)
    pretty_print_indels(ins_counts, del_counts)

    # Save: {epitran_phone: [[ref_phone, ratio], ...]}
    serialisable = {
        epi_ph: [[ref_ph, round(ratio, 6)] for ref_ph, ratio in pairs]
        for epi_ph, pairs in mapping.items()
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
    print(f"\nMapping saved to {args.output}")
    if args.threshold > 0:
        print(f"  (filtered at ratio >= {args.threshold})")


if __name__ == "__main__":
    main()
