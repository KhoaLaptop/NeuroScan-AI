import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import glob

# Each task draws every one of its classes from a single source dataset, so that
# source cannot predict label. Mixing sources across classes lets the network
# separate them by acquisition artefacts (resolution, framing) instead of anatomy.
TASKS = {
    'detection': {
        'class_to_idx': {'Normal': 0, 'tumor': 1},
        'aliases': {'no': 'Normal', 'yes': 'tumor'},
    },
    'subtype': {
        'class_to_idx': {'meningioma': 0, 'glioma': 1, 'pituitary tumor': 2},
        'aliases': {'pituitary': 'pituitary tumor'},
    },
}


class UnifiedDataset(Dataset):
    def __init__(self, root_dirs, transform=None, split='train', task='detection',
                 val_split=0.15, test_split=0.15, seed=42):
        """
        Args:
            root_dirs (list): List of root directories containing class folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): 'train', 'val', or 'test'.
            task (str): Which task's class mapping to use, see TASKS.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.
            seed (int): Random seed for splitting.
        """
        if task not in TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {list(TASKS)}.")

        self.transform = transform
        self.split = split
        self.task = task
        self.image_paths = []
        self.labels = []

        self.class_to_idx = TASKS[task]['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        aliases = TASKS[task]['aliases']

        all_images = []
        all_labels = []

        # Helper to normalize class names (handle spaces, aliases, etc.)
        def normalize_class_name(name):
            name = name.strip()
            return aliases.get(name, name)

        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                print(f"Warning: Directory {root_dir} does not exist.")
                continue
                
            # Walk through the directory
            for class_name in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                clean_name = normalize_class_name(class_name)
                
                if clean_name in self.class_to_idx:
                    label = self.class_to_idx[clean_name]
                    # Find all images
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        images = glob.glob(os.path.join(class_dir, ext))
                        for img_path in images:
                            all_images.append(img_path)
                            all_labels.append(label)
                else:
                    # print(f"Skipping unknown class folder: {class_name} in {root_dir}")
                    pass

        # Split data
        if not all_images:
            raise RuntimeError("No images found in the specified directories.")

        # First split: Train+Val vs Test
        # We want test_split portion for testing
        train_val_imgs, test_imgs, train_val_lbls, test_lbls = train_test_split(
            all_images, all_labels, test_size=test_split, stratify=all_labels, random_state=seed
        )

        # Second split: Train vs Val
        # We need to adjust val_split relative to the remaining data
        # If we want 15% val of TOTAL, and we removed 15% for test, we have 85% left.
        # 15% / 85% ~= 0.176
        relative_val_split = val_split / (1 - test_split)
        
        train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
            train_val_imgs, train_val_lbls, test_size=relative_val_split, stratify=train_val_lbls, random_state=seed
        )

        if self.split == 'train':
            self.image_paths = train_imgs
            self.labels = train_lbls
        elif self.split == 'val':
            self.image_paths = val_imgs
            self.labels = val_lbls
        elif self.split == 'test':
            self.image_paths = test_imgs
            self.labels = test_lbls
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle gracefully? 
            # For now, let's try to load the next one or raise
            raise e

        if self.transform:
            image = self.transform(image)

        return image, label
