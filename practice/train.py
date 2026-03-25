# train.py
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from clearml import Task, Dataset

class MnistImageDataset(TorchDataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.df = pd.read_csv(root_dir / "labels.csv")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root_dir / "images" / row["file_name"]).convert("L")
        x = self.transform(img)
        y = int(row["label"])
        return x, y

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

def main():
    task = Task.init(project_name="MNIST_CT", task_name="mnist_train_base")

    params = {
        "dataset_id": "e3071f1e80324335b54696fcb3813056",
        "epochs": 2,
        "batch_size": 128,
        "lr": 1e-3,
    }
    params = task.connect(params)

    if not params["dataset_id"]:
        raise ValueError("dataset_id가 비어 있습니다.")

    clearml_ds = Dataset.get(dataset_id=params["dataset_id"])
    data_dir = Path(clearml_ds.get_local_copy())

    ds = MnistImageDataset(data_dir)
    loader = DataLoader(ds, batch_size=int(params["batch_size"]), shuffle=True)

    model = SmallNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    criterion = nn.CrossEntropyLoss()

    logger = task.get_logger()

    for epoch in range(int(params["epochs"])):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        logger.report_scalar("loss", "train", avg_loss, epoch)
        logger.report_scalar("accuracy", "train", acc, epoch)
        print(f"epoch={epoch} loss={avg_loss:.4f} acc={acc:.4f}")

    task.upload_artifact("used_dataset_id", params["dataset_id"])

if __name__ == "__main__":
    main()