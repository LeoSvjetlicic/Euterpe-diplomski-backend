#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import json
from pathlib import Path

def detect_staff_area(img):
    """
    Detect the musical staff area and crop out text annotations
    """
    # Work with binary image for line detection
    binary_img = (img * 255).astype(np.uint8)
    
    # Invert for better line detection (black lines on white background)
    binary_img = 255 - binary_img
    
    # Create a kernel to detect long horizontal lines (staff lines)
    # Staff lines are typically very long and thin
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//3, 1))
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find horizontal projection to locate staff lines
    horizontal_sum = np.sum(horizontal_lines, axis=1)
    
    # Find staff line candidates - they should have significant horizontal content
    threshold = np.max(horizontal_sum) * 0.1
    staff_candidates = []
    
    for i, val in enumerate(horizontal_sum):
        if val > threshold:
            staff_candidates.append(i)
    
    if len(staff_candidates) < 3:  # Need at least 3 staff lines
        # Fallback: crop bottom 30% to remove chord letters
        crop_point = int(img.shape[0] * 0.7)
        return img[:crop_point, :]
    
    # Find the first and last staff lines across the entire image
    # Instead of grouping, find the absolute first and last staff lines
    first_staff_line = min(staff_candidates)
    last_staff_line = max(staff_candidates)
    
    # Add exactly 20 pixels padding as requested
    padding = 69
    
    staff_top = max(0, first_staff_line - padding)
    staff_bottom = min(img.shape[0], last_staff_line + padding)
    
    # Crop to staff area
    cropped_img = img[staff_top:staff_bottom, :]
    
    return cropped_img

def preprocess_image(image_path):
    """
    Preprocess image using the exact same steps as in your training code
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image in grayscale - exactly as in your code
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply binary threshold using Otsu's method - exactly as in your code
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Normalize to [0, 1] - exactly as in your code
    img = img.astype(np.float32) / 255.0
    
    # NEW: Detect and crop to staff area to remove text annotations
    img = detect_staff_area(img)
    
    # Resize maintaining aspect ratio, target height = 128 - exactly as in your code
    h, w = img.shape
    new_h = 128
    new_w = max(1, int(w * (new_h / h)))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def main():
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_image> <output_json>", file=sys.stderr)
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_json = sys.argv[2]
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(input_image)
        
        # Save the processed image for inspection
        debug_image_path = input_image + '_processed.png'
        
        # Convert back to 0-255 range for saving
        debug_img = (processed_img * 255).astype(np.uint8)
        
        # Use explicit PNG format to avoid extension issues
        success = cv2.imwrite(debug_image_path, debug_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if success:
            print(f"Processed image saved to: {debug_image_path}", file=sys.stderr)
        else:
            print(f"Warning: Could not save processed image to {debug_image_path}", file=sys.stderr)
        
        # Convert to format expected by the model
        h, w = processed_img.shape
        
        # Prepare output data
        result = {
            "data": processed_img.flatten().tolist(),
            "width": w,
            "height": h,
            "success": True
        }
        
        # Save to JSON file
        with open(output_json, 'w') as f:
            json.dump(result, f)
            
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        
        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }
        try:
            with open(output_json, 'w') as f:
                json.dump(error_result, f)
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()