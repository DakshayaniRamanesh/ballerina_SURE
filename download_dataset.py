from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor

REPO_ID = "phamquiluan/RCAEval"
DATA_DIR = Path("data")

DATA_DIR.mkdir(exist_ok=True)

api = HfApi()

print("=" * 60)
print("AEGIS - RCAEval Dataset Downloader")
print("=" * 60)

print("\nRetrieving repository file list...")

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"Total repository files: {len(files)}")

selected_files = [
    f for f in files
    if f.startswith("re1ob_")
    and f.endswith(
        ("metrics.parquet", "inject_time.txt")
    )
]

print(f"Selected files: {len(selected_files)}")

if not selected_files:
    print("\nFirst 30 repository files:")

    for f in files[:30]:
        print(f)

    raise RuntimeError(
        "No RE1-OB dataset files found."
    )

def download_file(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(DATA_DIR)
    )

with ThreadPoolExecutor(max_workers=4) as executor:
    for i, result in enumerate(
        executor.map(download_file, selected_files),
        start=1
    ):
        print(f"[{i}/{len(selected_files)}] {result}")

cases = [
    p for p in DATA_DIR.glob("re1ob_*")
    if p.is_dir()
    and (p / "metrics.parquet").exists()
    and (p / "inject_time.txt").exists()
]

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(f"Complete cases: {len(cases)}")
print(f"Expected cases: 125")
print(f"Dataset location: {DATA_DIR.resolve()}")

if len(cases) != 125:
    raise RuntimeError(
        "Unexpected number of complete cases. "
        "Inspect the dataset before training."
    )

print("SUCCESS: Dataset ready for training.")