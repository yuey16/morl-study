"""Download the pinned public Momba dependency once, before training."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent
    source = json.loads((root / "sources.json").read_text())
    repository = source["repository"].removeprefix("https://github.com/")
    for name, expected in source["files"].items():
        destination = root / "_deps" / "momba" / name
        if (
            destination.exists()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == expected
        ):
            continue
        url = (
            f"https://raw.githubusercontent.com/{repository}/{source['commit']}/{name}"
        )
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Source checksum mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print("Momba dependency ready.")


if __name__ == "__main__":
    main()
