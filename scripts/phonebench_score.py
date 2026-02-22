# python scripts/phonebench_score.py "Model & \ms{85.2}{1.3}"
import sys, re, argparse
import numpy as np

COUNTS = [4967, 6885, 287, 5497, 7762, 2500, 2400, 3985]
NAMES = ["DYS-ez", "DYS-ua", "CSD-us", "L1-eda", "L1-arc", "L2-so", "LID-fl", "GEO-v"]


def get_val(text):
    # Match \ms{MEAN}{STD} or raw number
    m = re.search(r"\\ms\s*\{([-\d\.]+)\}|([-\d\.]+)", text)
    return float(m.group(1) or m.group(2)) if m else None


def clean_name(text):
    return re.sub(r"\\[a-zA-Z]+\{|[\}\\]", "", text).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=str)
    args = parser.parse_args()
    row = args.rows.split("\n")
    TAKE_COL = [0, 1, 2, 3, 4, 5]
    print(
        "Computing weighted average scores... over columns:",
        [NAMES[i] for i in TAKE_COL],
    )
    print("===" * 20)

    for r in row:
        cells = r.split("&")
        if not cells:
            sys.exit(0)

        name = clean_name(cells[0])
        scores = [v for c in cells[1:] if (v := get_val(c)) is not None]
        scores = [scores[i] for i in TAKE_COL if i < len(scores)]

        if len(scores) == len(TAKE_COL):
            w_score = np.average(scores, weights=np.log([COUNTS[i] for i in TAKE_COL]))
            print(f"Model: {name} | Score: {w_score:.1f}")
        else:
            print(f"Model: {name} | Error: Found {len(scores)}/6 scores")
