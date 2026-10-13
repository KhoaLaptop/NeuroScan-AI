
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import sys
import os

from main import TASK_ROOT_DIRS
from src.dataset import UnifiedDataset
from src.model import get_model
from src.utils import evaluate_model

def test(task='detection'):
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Transforms (same as validation)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    root_dirs = TASK_ROOT_DIRS[task]

    print(f"Initializing Test Dataset for task '{task}'...")
    # Using default split ratios: val_split=0.15, test_split=0.15
    test_dataset = UnifiedDataset(root_dirs, transform=data_transforms, split='test', task=task)
    
    # Default num_workers to 0 on MacOS to avoid multiprocessing issues
    num_workers = 0 if sys.platform == 'darwin' else 4
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    
    num_classes = len(test_dataset.class_to_idx)
    class_names = [test_dataset.idx_to_class[i] for i in range(num_classes)]
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    print("Loading model...")
    model = get_model(num_classes=num_classes, pretrained=False) # Pretrained doesn't matter as we load weights
    
    model_path = f'best_model_{task}.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("Evaluating on Test Set...")
    evaluate_model(model, test_loader, device, class_names, output_path=f'confusion_matrix_{task}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detection', choices=list(TASK_ROOT_DIRS))
    args = parser.parse_args()
    test(args.task)
