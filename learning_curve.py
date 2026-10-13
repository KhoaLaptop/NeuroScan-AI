"""Does the detector need MORE Br35H data, or DIFFERENT data?

Trains stage 1 on growing fractions of Br35H and scores each model twice:
in-domain (Br35H val) and out-of-domain (hand-collected T1 multi-plane normals).
If out-of-domain accuracy stays flat while in-domain climbs, more Br35H will not
help and the gap is plane/sequence coverage, not volume.
"""

import csv
import glob
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from main import TASK_ROOT_DIRS, RandomResolutionJitter
from src.dataset import UnifiedDataset
from src.model import get_model
from src.train import train_model

FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
EPOCHS = 20
SEED = 42

# Whole normal cases, so every slice is reliably labelled Normal. The tumor folder
# is excluded on purpose: sampling showed many of its slices contain no tumor.
EXTERNAL_NORMAL_DIRS = [
    'Tumor Dataset/Normal',
    'Tumor Dataset/staged/normal-mri-brain-venogram',
    'Tumor Dataset/validation/normal-brain-mri-6',
]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def external_normal_accuracy(model, device):
    """Fraction of known-normal external slices the model calls Normal."""
    tf = eval_transform()
    paths = []
    for d in EXTERNAL_NORMAL_DIRS:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            paths += glob.glob(os.path.join(d, '**', ext), recursive=True)
    paths = sorted(set(paths))

    correct = 0
    total = 0
    model.eval()
    for p in paths:
        try:
            x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        except Exception:
            continue
        with torch.no_grad():
            pred = torch.argmax(F.softmax(model(x), dim=1), dim=1).item()
        correct += (pred == 0)  # index 0 == Normal
        total += 1
    return correct / total if total else 0.0, total


def main():
    device = get_device()
    print(f'device: {device}')

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        RandomResolutionJitter(),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    roots = TASK_ROOT_DIRS['detection']
    full_train = UnifiedDataset(roots, transform=train_tf, split='train', task='detection', val_split=0.2)
    val_set = UnifiedDataset(roots, transform=eval_transform(), split='val', task='detection', val_split=0.2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    order = list(range(len(full_train)))
    random.Random(SEED).shuffle(order)

    rows = []
    for frac in FRACTIONS:
        n = max(2, int(len(full_train) * frac))
        subset = Subset(full_train, order[:n])
        loaders = {
            'train': DataLoader(subset, batch_size=32, shuffle=True, num_workers=0),
            'val': val_loader,
        }

        print(f'\n===== fraction {frac} -> {n} training images =====', flush=True)
        model = get_model(num_classes=2, pretrained=True).to(device)

        labels = torch.tensor([full_train.labels[i] for i in order[:n]])
        counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
        weights = (counts.sum() / (2 * counts)).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model, hist = train_model(model, loaders, criterion, optimizer, device,
                                  num_epochs=EPOCHS,
                                  checkpoint_path=f'lc_tmp_{frac}.pth')

        in_domain = max(float(a) for a in hist)
        ext_acc, ext_n = external_normal_accuracy(model, device)
        rows.append({'fraction': frac, 'train_images': n,
                     'in_domain_val_acc': round(in_domain, 4),
                     'external_normal_acc': round(ext_acc, 4),
                     'external_n': ext_n})
        print(f'RESULT frac={frac} n={n} in_domain={in_domain:.4f} external={ext_acc:.4f}', flush=True)

        os.remove(f'lc_tmp_{frac}.pth')

    with open('learning_curve.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\n==================== LEARNING CURVE ====================')
    print(f'{"frac":>6} {"images":>7} {"Br35H val":>10} {"external normals":>18}')
    for r in rows:
        print(f'{r["fraction"]:>6} {r["train_images"]:>7} {r["in_domain_val_acc"]:>10.3f} {r["external_normal_acc"]:>18.3f}')
    first, last = rows[0], rows[-1]
    print(f'\nin-domain gain  {last["in_domain_val_acc"] - first["in_domain_val_acc"]:+.3f}')
    print(f'external gain   {last["external_normal_acc"] - first["external_normal_acc"]:+.3f}')
    print('\nIf external gain is near zero while in-domain climbs, more Br35H data')
    print('will not help - the gap is plane/sequence coverage, not volume.')
    print('Saved learning_curve.csv')


if __name__ == '__main__':
    main()
