"""
Download a subtree from a Modal Volume by explicitly listing and reading each
file via the SDK -- a workaround for `modal volume get` unreliably collapsing
multi-file directory downloads into a single file on some CLI versions.

Usage:
    python modal/download_outputs.py Qwen__Qwen3.5-4B/M1_emergent_misalignment/adapter
    python modal/download_outputs.py Qwen__Qwen3.5-4B/M1_emergent_misalignment/adapter --volume flavours-of-misalignment-results --local results
"""

import argparse
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download(remote_prefix: str, volume_name: str, local_root: Path):
    volume = modal.Volume.from_name(volume_name)

    local_dir = local_root / remote_prefix
    local_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for file in volume.listdir(remote_prefix):
        local_path = local_dir / file.path.split("/")[-1]
        with open(local_path, "wb") as f:
            for chunk in volume.read_file(file.path):
                f.write(chunk)
        print(f"downloaded {file.path} -> {local_path}")
        n += 1

    print(f"Downloaded {n} file(s) to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote_prefix", help="Path within the volume to download, e.g. Qwen__Qwen3.5-4B/M1_emergent_misalignment/adapter")
    parser.add_argument("--volume", default="flavours-of-misalignment-models")
    parser.add_argument("--local", default=str(PROJECT_ROOT / "models"))
    args = parser.parse_args()
    download(args.remote_prefix, args.volume, Path(args.local))
