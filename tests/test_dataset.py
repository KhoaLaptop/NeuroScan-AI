import sys
import os
import torch
from torchvision import transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest

from main import TASK_ROOT_DIRS
from src.dataset import UnifiedDataset

@pytest.mark.parametrize('task', ['detection', 'subtype'])
def test_dataset(task):
    root_dirs = TASK_ROOT_DIRS[task]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print(f"Initializing UnifiedDataset for task '{task}'...")
    dataset = UnifiedDataset(root_dirs=root_dirs, transform=transform, split='train', task=task)

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

    assert all(count > 0 for count in class_counts.values()), f"Empty class in {class_counts}"

    # Test getitem
    img, label = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label}")


if __name__ == "__main__":
    for t in ['detection', 'subtype']:
        test_dataset(t)
