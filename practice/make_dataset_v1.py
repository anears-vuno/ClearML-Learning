# make_dataset_v1.py
from pathlib import Path
import shutil
import pandas as pd
from torchvision.datasets import MNIST
from clearml import Dataset

ROOT = Path("mnist_data_v1")
IMG_DIR = ROOT / "images"

def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ds = MNIST(root="raw_data", train=True, download=True)

    rows = []
    for i in range(10_000):
        img, label = ds[i]
        img_path = IMG_DIR / f"{i:05d}.png"
        img.save(img_path)
        rows.append({"file_name": img_path.name, "label": int(label)})

    pd.DataFrame(rows).to_csv(ROOT / "labels.csv", index=False)

    dataset = Dataset.create(
        dataset_project="MNIST_CT",
        dataset_name="mnist_dataset",
    )
    dataset.add_files(str(ROOT))
    dataset.upload()
    dataset.finalize()

    print("Created dataset v1:", dataset.id)

if __name__ == "__main__":
    main()