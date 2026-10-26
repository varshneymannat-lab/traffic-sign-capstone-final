from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
PACKAGES = ["GTSRB_Final_Training_Images.zip", "GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_GT.zip"]


def main():
    parser = argparse.ArgumentParser(description="Download official GTSRB archives and inspect their labels.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/gtsrb/archives"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in PACKAGES:
        destination = args.output_dir / name
        if not destination.exists():
            partial = destination.with_suffix(".zip.partial")
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            with zipfile.ZipFile(partial) as archive:
                damaged = archive.testzip()
                if damaged:
                    raise ValueError(f"Damaged archive member: {damaged}")
            partial.rename(destination)
        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({"archive": name, "url": BASE_URL + name, "bytes": destination.stat().st_size, "sha256": digest.hexdigest()})
        print(f"Verified {name}", flush=True)
    class_counts = {}
    for split, name in [("official_train", PACKAGES[0]), ("official_test", PACKAGES[2])]:
        counts = Counter()
        with zipfile.ZipFile(args.output_dir / name) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                text = archive.read(member).decode("utf-8-sig")
                for row in csv.DictReader(io.StringIO(text), delimiter=";"):
                    if "ClassId" in row:
                        counts[int(row["ClassId"])] += 1
        class_counts[split] = dict(sorted(counts.items()))
    inventory = {"source": BASE_URL + "published-archive.html", "archives": records, "class_counts": class_counts}
    (args.output_dir / "download_manifest.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps({"class_counts": class_counts}, indent=2))


if __name__ == "__main__":
    main()
