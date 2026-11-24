from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("bayesian/outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cases = sorted(DATA_DIR.glob("re1ob_*"))

records = []
schemas = []

for case in cases:

    metrics_file = case / "metrics.parquet"
    injection_file = case / "inject_time.txt"

    if not metrics_file.exists():
        continue

    if not injection_file.exists():
        continue

    df = pd.read_parquet(metrics_file)

    injection_time = int(
        injection_file.read_text().strip()
    )

    _, service, fault, run = case.name.split("_")

    before = df[df["time"] < injection_time]
    after = df[df["time"] >= injection_time]

    schemas.append(set(df.columns))

    records.append({
        "case_id": case.name,
        "service": service,
        "fault": fault,
        "run": int(run),
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_timestamps": int(
            df["time"].duplicated().sum()
        ),
        "start_time": df["time"].min(),
        "end_time": df["time"].max(),
        "injection_time": injection_time,
        "before_injection": len(before),
        "after_injection": len(after),
        "injection_in_range": (
            df["time"].min()
            <= injection_time
            <= df["time"].max()
        )
    })

    print(
        f"{case.name}: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )

if not records:
    raise RuntimeError("No complete cases found")

audit = pd.DataFrame(records)

common_columns = set.intersection(*schemas)
all_columns = set.union(*schemas)

print("\n" + "=" * 60)
print("DATASET AUDIT")
print("=" * 60)

print("Total cases:", len(audit))
print("Total rows:", audit["rows"].sum())
print("Missing values:", audit["missing_values"].sum())

print("\nFault distribution:")
print(audit["fault"].value_counts())

print("\nService distribution:")
print(audit["service"].value_counts())

print("\nColumn count distribution:")
print(audit["columns"].value_counts())

print("\nInvalid injection timestamps:")
print(audit[~audit["injection_in_range"]])

print("\nCommon columns:", len(common_columns))

for column in sorted(common_columns):
    print(column)

print("\nNon-common columns:")

for column in sorted(all_columns - common_columns):
    print(column)

audit.to_csv(
    OUTPUT_DIR / "dataset_audit.csv",
    index=False
)

(OUTPUT_DIR / "common_columns.txt").write_text(
    "\n".join(sorted(common_columns)),
    encoding="utf-8"
)

print("\nAudit completed successfully.")