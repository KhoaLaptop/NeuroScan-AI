import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import os
import sys

from src.dataset import UnifiedDataset
from src.model import get_model
from src.train import train_model
from src.utils import evaluate_model

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
            transforms.Resize((224, 224)),
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

    root_dirs = [
        '/Users/khoapc/Documents/Project Brain Classifier/data/train',
        '/Users/khoapc/Documents/Project Brain Classifier/data/val',
        '/Users/khoapc/Documents/Project Brain Classifier/brain_tumor_dataset_3',
        '/Users/khoapc/Documents/Project Brain Classifier/Stroke_classification'
    ]

    print("Initializing Datasets...")
    image_datasets = {
        'train': UnifiedDataset(root_dirs, transform=data_transforms['train'], split='train', val_split=0.2),
        'val': UnifiedDataset(root_dirs, transform=data_transforms['val'], split='val', val_split=0.2)
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    class_names = [image_datasets['train'].idx_to_class[i] for i in range(10)]
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")

    model = get_model(num_classes=10, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Optional

    print("Starting training...")
    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=args.epochs)

    print("Saving model...")
    torch.save(model.state_dict(), 'brain_disease_mobilenet.pth')
    print("Model saved to brain_disease_mobilenet.pth")

    print("Evaluating model...")
    evaluate_model(model, dataloaders['val'], device, class_names)

if __name__ == "__main__":
    # Default num_workers to 0 on MacOS to avoid multiprocessing issues
    default_num_workers = 0 if sys.platform == 'darwin' else 4
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=default_num_workers, help='Number of dataloader workers')
    args = parser.parse_args()
    
    main(args)
