"""Download the small pretrained MobileNet-SSD model from its source repository."""

from pathlib import Path
from urllib.request import urlopen


BASE = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
FILES = {
    "MobileNetSSD_deploy.prototxt": "deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "mobilenet_iter_73000.caffemodel",
}


def main() -> None:
    destination = Path(__file__).parent / "models"
    destination.mkdir(exist_ok=True)
    for local_name, remote_name in FILES.items():
        target = destination / local_name
        print(f"Downloading {local_name} ...")
        with urlopen(BASE + remote_name, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"Saved {target} ({target.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
