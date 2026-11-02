"""Correct the five-week offset of the Quality series in data_input/factor_returns.csv.

In the file as received, the weekly Quality return stamped at week t is the return earned in week t + 5:
  - with Momentum and Value it correlates -0.04 and -0.03 at the same date, but 0.78 and 0.84 when taken
    five weeks earlier, in every three-year block from 2007 to 2025, and at no other lag (|corr| <= 0.4);
  - summed to months it matches the Quality series of factor_benchmark_returns.xlsx only after the same
    shift (0.80, against 0.01), while Momentum and Value match without one (0.86, 0.87).

The correction moves Quality five weeks later and drops the first five weeks, whose Quality return is not
in the file. Momentum and Value are unchanged. The file as received is kept as factor_returns_raw.csv;
running the script again rebuilds factor_returns.csv from it.

Usage: python fix_quality_offset.py
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data_input"
RAW, OUT = DATA / "factor_returns_raw.csv", DATA / "factor_returns.csv"
OFFSET = 5


def offset_evidence(df):
    """Correlation of Quality with the average of Momentum and Value, at the same date and OFFSET weeks earlier."""
    mv = (df.Momentum + df.Value) / 2
    return mv.corr(df.Quality), mv.corr(df.Quality.shift(OFFSET))


if not RAW.exists():
    current = pd.read_csv(OUT, parse_dates=["Date"]).set_index("Date")
    same, shifted = offset_evidence(current)
    assert abs(same) < 0.1 and shifted > 0.7, f"{OUT.name} does not look like the file as received ({same:.2f}, {shifted:.2f})"
    current.to_csv(RAW)
raw = pd.read_csv(RAW, parse_dates=["Date"]).set_index("Date")
same, shifted = offset_evidence(raw)
assert abs(same) < 0.1 and shifted > 0.7, (same, shifted)

fixed = raw.copy()
fixed["Quality"] = raw["Quality"].shift(OFFSET)
fixed = fixed.iloc[OFFSET:]
assert fixed.notna().all().all()
assert offset_evidence(fixed)[0] > 0.7
fixed.to_csv(OUT)
print(f"{OUT.name}: Quality moved {OFFSET} weeks later; {len(fixed)} weeks from {fixed.index[0].date()} to "
      f"{fixed.index[-1].date()} (correlation with Momentum/Value {same:.2f} before, {offset_evidence(fixed)[0]:.2f} after)")
