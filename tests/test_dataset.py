import sys
import os
import torch
from torchvision import transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import UnifiedDataset

def test_dataset():
    root_dirs = [
        '/Users/khoapc/Documents/Project Brain Classifier/data/train',
        '/Users/khoapc/Documents/Project Brain Classifier/data/val',
        '/Users/khoapc/Documents/Project Brain Classifier/brain_tumor_dataset_3',
        '/Users/khoapc/Documents/Project Brain Classifier/Stroke_classification'
    ]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("Initializing UnifiedDataset...")
    dataset = UnifiedDataset(root_dirs=root_dirs, transform=transform, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.class_to_idx}")

    # Check if we have samples for all classes
    class_counts = {k: 0 for k in dataset.class_to_idx.keys()}
    
    for _, label in dataset:
        class_name = dataset.idx_to_class[label]
        class_counts[class_name] += 1
    
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
        if count == 0:
            print(f"WARNING: No samples found for class {cls}!")

    # Test getitem
    img, label = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")

if __name__ == "__main__":
    test_dataset()
