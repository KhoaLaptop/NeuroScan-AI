
import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import torch.nn.functional as F

from src.model import get_model

def predict(image_path, model_path='best_model.pth'):
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Define class mapping (must match training)
    class_names = [
        'meningioma',
        'glioma',
        'pituitary tumor',
        'MildDemented',
        'ModerateDemented',
        'NonDemented',
        'VeryMildDemented',
        'Haemorrhagic',
        'Ischemic',
        'Normal'
    ]

    # Transforms (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension
        input_tensor = input_tensor.to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model = get_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_score:.4f}")

    return predicted_class, confidence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain disease from image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the model file')
    args = parser.parse_args()

    predict(args.image_path, args.model)
