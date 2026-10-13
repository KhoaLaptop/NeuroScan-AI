
import torch
from torchvision import transforms
from PIL import Image
import argparse
import csv
import glob
import os
import torch.nn.functional as F

from src.model import get_model

DETECTION_CLASSES = ['Normal', 'tumor']
SUBTYPE_CLASSES = ['meningioma', 'glioma', 'pituitary tumor']

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, num_classes, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Train it first.")
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def classify(model, tensor, class_names):
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    return class_names[predicted_idx.item()], confidence.item()


def predict_image(detector, subtyper, image_path, transform, device):
    """Stage 1 decides tumor vs Normal; only tumors go on to subtyping."""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    detection, detection_conf = classify(detector, tensor, DETECTION_CLASSES)
    if detection == 'Normal':
        return 'Normal', detection_conf, ''

    subtype, subtype_conf = classify(subtyper, tensor, SUBTYPE_CLASSES)
    return subtype, detection_conf, f'{subtype_conf:.4f}'


def find_images(folder_path):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    return sorted(images)


def predict(image_path, detector_path='best_model_detection.pth',
            subtyper_path='best_model_subtype.pth', output_csv='predictions.csv'):
    device = get_device()
    print(f"Using device: {device}")

    detector = load_model(detector_path, len(DETECTION_CLASSES), device)
    subtyper = load_model(subtyper_path, len(SUBTYPE_CLASSES), device)
    transform = get_transform()

    if os.path.isdir(image_path):
        image_files = find_images(image_path)
        if not image_files:
            print(f"No images found in '{image_path}'.")
            return

        print(f"Found {len(image_files)} images. Running predictions...")
        results = []
        for path in image_files:
            try:
                prediction, detection_conf, subtype_conf = predict_image(detector, subtyper, path, transform, device)
                results.append((path, prediction, f'{detection_conf:.4f}', subtype_conf))
                print(f"{path}: {prediction} (detection {detection_conf:.4f}"
                      + (f", subtype {subtype_conf}" if subtype_conf else "") + ")")
            except Exception as e:
                print(f"Error processing '{path}': {e}")

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'prediction', 'detection_confidence', 'subtype_confidence'])
            writer.writerows(results)
        print(f"\nSaved {len(results)} predictions to {output_csv}")

        print("\nClass distribution:")
        for class_name in ['Normal'] + SUBTYPE_CLASSES:
            count = sum(1 for _, pred, _, _ in results if pred == class_name)
            print(f"{class_name}: {count}")

        return results

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    prediction, detection_conf, subtype_conf = predict_image(detector, subtyper, image_path, transform, device)
    print(f"Prediction: {prediction}")
    print(f"Detection confidence: {detection_conf:.4f}")
    if subtype_conf:
        print(f"Subtype confidence: {subtype_conf}")
    return prediction, detection_conf, subtype_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain tumor class from an image or a folder of images')
    parser.add_argument('image_path', type=str, help='Path to an image file or a folder of images')
    parser.add_argument('--detector', type=str, default='best_model_detection.pth', help='Stage 1 model (tumor vs Normal)')
    parser.add_argument('--subtyper', type=str, default='best_model_subtype.pth', help='Stage 2 model (tumor subtype)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='CSV path for batch results (used when image_path is a folder)')
    args = parser.parse_args()

    predict(args.image_path, args.detector, args.subtyper, args.output)
