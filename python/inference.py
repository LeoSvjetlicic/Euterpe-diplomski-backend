#!/usr/bin/env python3
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def ctc_decode(token_list):
    """
    Collapse consecutive duplicates and remove <BLANK> tokens for CTC decoding.
    """
    decoded = []
    prev_token = None
    for token in token_list:
        if token != prev_token and token != "<BLANK>":
            decoded.append(token)
        prev_token = token
    return decoded

def load_model():
    """Load the actual CRNN model"""
    import torch.nn as nn
    
    # Define your exact CRNN class
    class CRNN(nn.Module):
        def __init__(self, vocab_size):
            super(CRNN, self).__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 

                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 

                nn.Conv2d(64, 128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 

                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
            )

            # Ovo mi nije jasnoooo
            self.rnn_input_size = 256 * 6
            self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

            self.fc = nn.Linear(256 * 2, vocab_size)

        def forward(self, x):
            x = self.cnn(x)  # Shape: (B, C, H=8, W/16)
            b, c, h, w = x.size()
            x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
            x = x.contiguous().view(b, w, c * h)  # (B, W, C*H) → sequence length W
            x, _ = self.rnn(x)  # output shape: (B, W, 512)
            x = self.fc(x)      # output shape: (B, W, vocab_size)
            return x
    
    # Load vocabulary to get vocab_size
    vocab_path = Path(__file__).parent.parent / "vocab.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "crnn_epoch_19.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model and load weights
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CRNN(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def preprocess_input(data, width, height):
    """Convert input data to tensor format"""
    # Reshape flat array to image dimensions
    expected_size = height * width
    actual_size = len(data)
    
    if expected_size != actual_size:
        print(f"Warning: Expected {expected_size} pixels but got {actual_size}", file=sys.stderr)
        # Crop or pad data to match expected dimensions
        if actual_size > expected_size:
            data = data[:expected_size]
        else:
            data = data + [0] * (expected_size - actual_size)
    
    image_array = np.array(data, dtype=np.float32).reshape(height, width)
    
    # Add batch and channel dimensions
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return tensor

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <input_file>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Extract image data
        data = input_data['data']
        width = input_data['width']
        height = input_data['height']
        
        # Preprocess input
        input_tensor = preprocess_input(data, width, height)
        
        # Load model
        model = load_model()
        
        # Run inference
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get predictions using argmax
        output_indices = output.argmax(dim=2)[0].cpu().numpy()
        
        # Load vocabulary to create idx_to_char mapping
        vocab_path = Path(__file__).parent.parent / "vocab.json"
        with open(vocab_path, 'r') as f:
            token_to_idx = json.load(f)
        
        # Create idx_to_char mapping
        idx_to_char = {idx: token for token, idx in token_to_idx.items()}
        idx_to_char[0] = "<BLANK>"
        
        predicted_chars = [idx_to_char.get(idx, "<UNK>") for idx in output_indices]
        
        # Decode CTC output
        decoded_tokens = ctc_decode(predicted_chars)
        
        # Convert token names back to indices for compatibility with server
        if decoded_tokens:
            token_indices = [token_to_idx.get(token, 0) for token in decoded_tokens]
        else:
            token_indices = []
        
        # Prepare output
        result = {
            "tokens": token_indices,
            "logits_shape": list(output.shape),
            "success": True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "success": False
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()