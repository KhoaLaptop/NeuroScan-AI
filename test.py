
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os

from src.dataset import UnifiedDataset
from src.model import get_model
from src.utils import evaluate_model

def test():
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

    root_dirs = [
        '/Users/khoapc/Documents/Project Brain Classifier/data/train',
        '/Users/khoapc/Documents/Project Brain Classifier/data/val',
        '/Users/khoapc/Documents/Project Brain Classifier/brain_tumor_dataset_3',
        '/Users/khoapc/Documents/Project Brain Classifier/Stroke_classification'
    ]

    print("Initializing Test Dataset...")
    # Using default split ratios: val_split=0.15, test_split=0.15
    test_dataset = UnifiedDataset(root_dirs, transform=data_transforms, split='test')
    
    # Default num_workers to 0 on MacOS to avoid multiprocessing issues
    num_workers = 0 if sys.platform == 'darwin' else 4
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    
    class_names = [test_dataset.idx_to_class[i] for i in range(10)]
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    print("Loading model...")
    model = get_model(num_classes=10, pretrained=False) # Pretrained doesn't matter as we load weights
    
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("Evaluating on Test Set...")
    evaluate_model(model, test_loader, device, class_names)

if __name__ == "__main__":
    test()
