import os
import glob
import random
import numpy as np
from typing import Sequence, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model_4d import Model4D

class MRLEyeDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, exts=(".png", ".jpg", ".jpeg")):
        self.root_dir = root_dir
        self.transform = transform
        self.images: List[str] = []
        self.labels: List[int] = []

        for subdir in sorted(glob.glob(os.path.join(root_dir, '[sS]*'))):
            if not os.path.isdir(subdir):
                continue
            for img_path in glob.glob(os.path.join(subdir, '*')):
                if not img_path.lower().endswith(exts):
                    continue
                fname = os.path.basename(img_path)
                fname_no_ext = fname.rsplit('.', 1)[0] if '.' in fname else fname
                parts = fname_no_ext.split('_')
                # Robust parsing for short/full formats
                if len(parts) < 5:
                    print(f"Warning: Skipping short filename {fname} (parts: {len(parts)})")
                    continue
                eye_state_str = None
                if len(parts) >= 8 and len(parts[1]) == 5 and parts[
                    1].isdigit(): # Full format: [1] is 5-digit image ID
                    eye_state_str = parts[4]
                else: # Short format: eye_state at [3]
                    eye_state_str = parts[3]
                if eye_state_str not in ("0", "1"):
                    print(f"Warning: Invalid eye_state '{eye_state_str}' in {fname}")
                    continue
                self.images.append(img_path)
                self.labels.append(int(eye_state_str))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Check path and extensions.")
        num_open = sum(self.labels)
        print(f"Loaded {len(self.images)} images: {num_open} open (1), {len(self.labels) - num_open} closed (0)")
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        p = self.images[idx]
        y = self.labels[idx]
        img = Image.open(p).convert('L') # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, y

# Training / Eval
def train_one_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        device: str = "cuda"
) -> Tuple[List[float], List[float]]:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    train_losses, val_accs = [], []
    for ep in range(1, epochs + 1):
        # train
        model.train()
        run_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        train_losses.append(run_loss / max(1, len(train_loader)))
        # validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).int().cpu().numpy()
                preds.extend(pred.tolist())
                trues.extend(labels.cpu().numpy().astype(int).tolist())
        acc = accuracy_score(trues, preds)
        val_accs.append(acc)
        scheduler.step(acc)
        print(f"Epoch {ep:02d}/{epochs} | TrainLoss {train_losses[-1]:.4f} | ValAcc {acc:.4f}")
    return train_losses, val_accs
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda") -> Tuple[float, float, float]:
    model.eval().to(device)
    preds, trues = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).int().cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend(labels.cpu().numpy().astype(int).tolist())
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    return acc, prec, rec
# Utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def make_loaders(
        ds: MRLEyeDataset,
        batch_size: int = 32,
        num_workers: int = 2,
        val_split: float = 0.1,
        test_split: float = 0.2,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Sequence[int], Sequence[int], Sequence[int]]:
    idx_all = np.arange(len(ds))
    y_all = np.array(ds.labels)
    # first train+val vs test
    idx_trv, idx_te = train_test_split(
        idx_all, test_size=test_split, random_state=seed, stratify=y_all
    )
    # then train vs val
    y_trv = y_all[idx_trv]
    val_ratio = val_split / (1.0 - test_split)
    idx_tr, idx_val = train_test_split(
        idx_trv, test_size=val_ratio, random_state=seed, stratify=y_trv
    )
    train_loader = DataLoader(Subset(ds, idx_tr), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(ds, idx_val), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(ds, idx_te), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, idx_tr, idx_val, idx_te

def main():
    root_dir = r"D:\Download\pyCharmpPro\sleep_detect\archive\mrlEyes_2018_01"
    epochs = 30
    bs = 32
    workers = 2
    seed = 42
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Transforms
    # 4D (24x24, grayscale 1ch)
    tf_24 = T.Compose([
        T.Resize((24, 24)),
        T.RandomRotation(10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomResizedCrop(24, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    # Datasets & loaders
    ds_24 = MRLEyeDataset(root_dir, transform=tf_24)
    tr_24, va_24, te_24, _, _, _ = make_loaders(ds_24, batch_size=bs, num_workers=workers)
    results = {}
    # Train 4D (check if exists)
    model_path_4d = "model_4d.pth"
    if os.path.exists(model_path_4d):
        print(f"\nSkipping 4D Training (model exists: {model_path_4d})")
        model_4d = Model4D()
        model_4d.load_state_dict(torch.load(model_path_4d, map_location=device))
    else:
        print("\nTraining 4D (24x24)")
        model_4d = Model4D()
        train_one_model(model_4d, tr_24, va_24, epochs=epochs, device=device)
        torch.save(model_4d.state_dict(), model_path_4d)
    acc, prec, rec = evaluate(model_4d, te_24, device)
    results["4D"] = (acc, prec, rec)
    print(f"4D Test -> Acc: {acc:.4f} Prec: {prec:.4f} Rec: {rec:.4f}")
    # Summary
    print("\nModel Comparison:")
    print("| Model | Accuracy | Precision | Recall |")
    print("|--------|----------|-----------|--------|")
    for name, (acc, prec, rec) in results.items():
        print(f"| {name:<6} | {acc:.4f} | {prec:.4f} | {rec:.4f} |")
if __name__ == "__main__":
    main()