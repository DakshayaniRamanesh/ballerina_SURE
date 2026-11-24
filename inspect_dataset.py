from pathlib import Path
import pandas as pd

data_dir = Path("data")
cases = sorted(data_dir.glob("re1ob_*"))

print("Number of cases:", len(cases))

if not cases:
    raise RuntimeError("Dataset not found")

case = cases[0]

df = pd.read_parquet(case / "metrics.parquet")

print("\nCase:", case.name)
print("\nDataset shape:", df.shape)

print("\nColumn names:")
for col in df.columns:
    print(col)

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFault injection time:")
print((case / "inject_time.txt").read_text())