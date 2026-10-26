"""Download the MobileSAM weights used for automatic roof detection.

    python scripts/download_sam.py

Source: the official MobileSAM repository (github.com/ChaoningZhang/MobileSAM).
The file is checked against a known SHA-256 before it is kept.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
DEST = Path(__file__).resolve().parents[1] / "models" / "sam" / "mobile_sam.pt"


def main() -> int:
    if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print(f"already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != SHA256:
        tmp.unlink()
        print(f"checksum mismatch ({digest}); file discarded", file=sys.stderr)
        return 1
    tmp.replace(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
