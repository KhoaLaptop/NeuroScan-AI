"""Fine-tune the stage 1 detector on Br35H plus hand-collected T1 multi-plane data,
then compare old vs new on held-out external scans.

Held out (never trained on):
  - validation/normal-brain-mri-6  (normal case; Axial_T1/ and Sagittal_T1/ are copies of it)
  - 320x320 meningioma slices      (look like a separate tumor patient)
"""

import csv
import glob
import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from main import TASK_ROOT_DIRS, RandomResolutionJitter
from src.dataset import UnifiedDataset
from src.model import get_model
from src.train import train_model

OLD_MODEL = 'best_model_detection.pth'
NEW_MODEL = os.environ.get('NEW_MODEL', 'best_model_detection_v2.pth')
EPOCHS = 15
D = 'Tumor Dataset'
NORMAL, TUMOR = 0, 1

NEW_NORMAL_DIRS = [f'{D}/Normal', f'{D}/staged/normal-mri-brain-venogram']
TUMOR_DIRS = [f'{D}/staged/meningioma_case-unknown', f'{D}/tumor']
# Explicit file list, not image size: later tumor batches also contain 320x320 slices.
HOLDOUT_LIST = f'{D}/holdout_tumor_slice_labels.csv'

TESTS = [
    ('Axial T1 (validation case)', 'Axial_T1', NORMAL),
    ('Sagittal T1 (validation case)', 'Sagittal_T1', NORMAL),
    ('Full validation case', f'{D}/validation/normal-brain-mri-6', NORMAL),
]


def images_in(folder):
    out = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
        out += glob.glob(os.path.join(folder, '**', ext), recursive=True)
    return sorted(set(out))


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.labels = [label for _, label in items]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert('RGB')), label


def build_new_data():
    normals = [p for d in NEW_NORMAL_DIRS for p in images_in(d)]
    with open(HOLDOUT_LIST) as f:
        tumors_holdout = sorted(r['path'] for r in csv.DictReader(f))
    held = set(tumors_holdout)
    tumors_train = [p for d in TUMOR_DIRS for p in images_in(d) if p not in held]

    train_items = [(p, NORMAL) for p in normals] + [(p, TUMOR) for p in tumors_train]

    test_sets = [(name, images_in(folder), label) for name, folder, label in TESTS]
    test_sets.append(('Held-out tumor case (320px)', tumors_holdout, TUMOR))

    # The project's recurring failure is test images leaking into training.
    train_hashes = {md5(p) for p, _ in train_items}
    for name, paths, _ in test_sets:
        leaked = [p for p in paths if md5(p) in train_hashes]
        assert not leaked, f'{len(leaked)} test images from "{name}" are also in training'

    return train_items, test_sets


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def eval_tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def accuracy(model, paths, label, device):
    tf = eval_tf()
    correct = 0
    model.eval()
    for p in paths:
        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            correct += int(torch.argmax(model(x), dim=1).item() == label)
    return correct / len(paths) if paths else float('nan')


def load(path, device):
    m = get_model(num_classes=2, pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)


def main():
    device = get_device()
    print(f'device: {device}', flush=True)

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
    br35h_train = UnifiedDataset(roots, transform=train_tf, split='train', task='detection', val_split=0.2)
    br35h_val = UnifiedDataset(roots, transform=eval_tf(), split='val', task='detection', val_split=0.2)
    br35h_test = UnifiedDataset(roots, transform=None, split='test', task='detection')

    new_items, test_sets = build_new_data()
    # Repeat the scarcer class so the hand-collected data is roughly balanced;
    # otherwise the model learns "T1/T2 multi-plane look -> Normal" (seen in v4).
    n_norm = sum(1 for _, l in new_items if l == NORMAL)
    tumors = [it for it in new_items if it[1] == TUMOR]
    repeat = max(1, round(n_norm / max(1, len(tumors))))
    new_items = new_items + tumors * (repeat - 1)
    print(f'new data: {n_norm} normal, {len(tumors)} tumor x{repeat}', flush=True)
    new_train = ListDataset(new_items, train_tf)
    combined = ConcatDataset([br35h_train, new_train])
    print(f'training: {len(br35h_train)} Br35H + {len(new_train)} new = {len(combined)}', flush=True)

    labels = torch.tensor(br35h_train.labels + new_train.labels)
    counts = torch.bincount(labels, minlength=2).float()
    weights = (counts.sum() / (2 * counts)).to(device)
    print(f'class counts (Normal, tumor): {counts.tolist()}', flush=True)

    model = load(OLD_MODEL, device)
    loaders = {
        'train': DataLoader(combined, batch_size=32, shuffle=True, num_workers=0),
        'val': DataLoader(br35h_val, batch_size=32, shuffle=False, num_workers=0),
    }
    train_model(model, loaders, nn.CrossEntropyLoss(weight=weights),
                optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
                device, num_epochs=EPOCHS, checkpoint_path=NEW_MODEL)

    old, new = load(OLD_MODEL, device), load(NEW_MODEL, device)
    br35h_test_sets = [
        (f'Br35H test — {name}', [p for p, l in zip(br35h_test.image_paths, br35h_test.labels) if l == lbl], lbl)
        for name, lbl in (('Normal', NORMAL), ('tumor', TUMOR))
    ]

    rows = []
    for name, paths, label in br35h_test_sets + test_sets:
        rows.append({'test': name, 'n': len(paths), 'truth': 'Normal' if label == NORMAL else 'tumor',
                     'old': round(accuracy(old, paths, label, device), 3),
                     'new': round(accuracy(new, paths, label, device), 3)})

    with open(NEW_MODEL.replace('.pth', '_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\n===================== OLD vs NEW =====================')
    print(f'{"test":38} {"n":>4} {"truth":>7} {"old":>6} {"new":>6} {"change":>7}')
    for r in rows:
        print(f'{r["test"]:38} {r["n"]:>4} {r["truth"]:>7} {r["old"]:>6.3f} {r["new"]:>6.3f} {r["new"] - r["old"]:>+7.3f}')
    print('Saved', NEW_MODEL.replace('.pth', '_results.csv'), 'and', NEW_MODEL)


if __name__ == '__main__':
    main()
