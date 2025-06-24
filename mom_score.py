import pandas as pd
import numpy as np
from pathlib import Path

# ======================
# 1. DATA LOADING (WITH CHECKS)
# ======================
input_path = "stock_prices.xlsx"

# Validate file exists
if not Path(input_path).exists():
    raise FileNotFoundError(f"File not found: {input_path}")

try:
    df = pd.read_excel(input_path, index_col="Date")
    print("Data loaded successfully. Shape:", df.shape)
    print("Sample data:\n", df.head(2))
except Exception as e:
    print("Error loading data:", str(e))
    raise

# ======================
# 2. DATA VALIDATION
# ======================
if df.empty:
    raise ValueError("Loaded DataFrame is empty")

# Check index is datetime
try:
    df.index = pd.to_datetime(df.index)
    print("Date index converted to datetime")
except Exception as e:
    print("Error converting index to datetime:", str(e))
    raise

# ======================
# 3. RETURN CALCULATION (DEBUGGED)
# ======================
returns = df.pct_change()

# Identify FIRST NaN in each column (delisting)
delisting_points = returns.isna().idxmax()

# Set only the FIRST NaN to -100%, keep others as NaN
for stock in returns.columns:
    if pd.notna(delisting_points[stock]):
        returns.loc[delisting_points[stock], stock] = -1.0

print("\nReturns after delisting adjustment:")
print(returns.tail())

# ======================
# 4. MOMENTUM CALCULATION
# ======================
def safe_rolling_prod(series, window):
    """Handles NaN values correctly in rolling window"""
    return series.rolling(window, min_periods=1).apply(
        lambda x: np.prod(1 + x) - 1, 
        raw=False
    )

momentum_6m = returns.apply(lambda x: safe_rolling_prod(x, 126)).shift(1)
momentum_12m = returns.apply(lambda x: safe_rolling_prod(x, 252)).shift(1)

print("\n6M Momentum sample:\n", momentum_6m.tail())
print("\n12M Momentum sample:\n", momentum_12m.tail())

# ======================
# 5. SAVING WITH VERIFICATION
# ======================
output_path = "momentum_score.xlsx"

try:
    with pd.ExcelWriter(output_path) as writer:
        returns.to_excel(writer, sheet_name="Returns")
        momentum_6m.to_excel(writer, sheet_name="6M_Momentum")
        momentum_12m.to_excel(writer, sheet_name="12M_Momentum")
    
    # Verify file was created
    if Path(output_path).exists():
        print(f"\nSuccess! File saved to {output_path}")
        print("File size:", Path(output_path).stat().st_size, "bytes")
    else:
        raise IOError("File was not created")
except Exception as e:
    print("Error saving file:", str(e))
    raise