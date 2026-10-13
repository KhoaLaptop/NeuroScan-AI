import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import os
import random
import sys

from src.dataset import UnifiedDataset
from src.model import get_model
from src.train import train_model
from src.utils import evaluate_model


BR35H = '/Users/khoapc/.cache/kagglehub/datasets/ahmedhamada0/brain-tumor-detection/versions/12'
BRAIN_TUMOR_MRI = '/Users/khoapc/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2'

# Stage 1 uses Br35H alone (its yes/ and no/ come from one pipeline); stage 2 uses
# the figshare-derived tumor classes alone. Neither stage lets source predict label.
TASK_ROOT_DIRS = {
    'detection': [BR35H],
    'subtype': [f'{BRAIN_TUMOR_MRI}/Training', f'{BRAIN_TUMOR_MRI}/Testing'],
}


class RandomResolutionJitter:
    """Downsample then upsample back, so sharpness cannot act as a class cue.

    The tumor classes come from figshare (uniform 512x512) and Normal from Br35H
    (mostly ~225x225 web images); without this the network separates the classes
    by source resolution instead of anatomy.
    """

    def __init__(self, min_size=96, max_size=224, p=0.8):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        size = random.randint(self.min_size, self.max_size)
        return img.resize((size, size), Image.BICUBIC).resize((224, 224), Image.BICUBIC)

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            RandomResolutionJitter(),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    root_dirs = TASK_ROOT_DIRS[args.task]
    checkpoint_path = f'best_model_{args.task}.pth'

    print(f"Initializing Datasets for task '{args.task}'...")
    image_datasets = {
        'train': UnifiedDataset(root_dirs, transform=data_transforms['train'], split='train', task=args.task, val_split=0.2),
        'val': UnifiedDataset(root_dirs, transform=data_transforms['val'], split='val', task=args.task, val_split=0.2)
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    num_classes = len(image_datasets['train'].class_to_idx)
    class_names = [image_datasets['train'].idx_to_class[i] for i in range(num_classes)]
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")

    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Classes are imbalanced (e.g. far fewer Normal images than tumor images),
    # so weight the loss inversely to class frequency in the training set.
    train_labels = torch.tensor(image_datasets['train'].labels)
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = (class_counts.sum() / (num_classes * class_counts)).to(device)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Optional

    print("Starting training...")
    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, device,
                                         num_epochs=args.epochs, checkpoint_path=checkpoint_path)

    print(f"Best model saved to {checkpoint_path}")

    print("Evaluating model...")
    evaluate_model(model, dataloaders['val'], device, class_names)

if __name__ == "__main__":
    # Default num_workers to 0 on MacOS to avoid multiprocessing issues
    default_num_workers = 0 if sys.platform == 'darwin' else 4
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detection', choices=list(TASK_ROOT_DIRS), help='Which stage to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=default_num_workers, help='Number of dataloader workers')
    args = parser.parse_args()
    
    main(args)
