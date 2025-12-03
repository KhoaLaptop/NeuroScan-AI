import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import get_model

def test_model():
    print("Initializing Model...")
    model = get_model(num_classes=10, pretrained=False) # No need to download weights for shape check
    
    print("Model initialized.")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 10), f"Expected output shape {(batch_size, 10)}, got {output.shape}"
    print("Test passed!")

if __name__ == "__main__":
    test_model()
