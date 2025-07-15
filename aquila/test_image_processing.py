#!/usr/bin/env python3
"""
Test script to verify image processing fixes
"""
import os
import sys
import asyncio
from pathlib import Path
from PIL import Image
import base64
import json

# Add the current directory to sys.path so we can import from server.py
sys.path.insert(0, '.')

# Test the image extraction function
def test_image_extraction():
    """Test PDF image extraction with format conversion"""
    print("Testing PDF image extraction...")
    
    # Import the function from server.py
    from server import extract_images_from_pdf
    
    # Test with available PDF
    pdf_path = "test_maintenance_manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file {pdf_path} not found")
        return False
    
    try:
        images = extract_images_from_pdf(pdf_path)
        print(f"✅ Extracted {len(images)} images")
        
        # Test each extracted image
        for i, image_path in enumerate(images):
            print(f"Testing image {i+1}: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"❌ Image file {image_path} not found")
                continue
            
            # Try to open with PIL
            try:
                with Image.open(image_path) as img:
                    print(f"✅ Image format: {img.format}, Mode: {img.mode}, Size: {img.size}")
                    
                    # Check if it's a supported format
                    if img.format in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                        print(f"✅ Image format {img.format} is supported by OpenAI")
                    else:
                        print(f"❌ Image format {img.format} is NOT supported by OpenAI")
                    
                    # Try to convert to base64 (what OpenAI expects)
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        print(f"✅ Base64 conversion successful (length: {len(base64_image)})")
                        
            except Exception as e:
                print(f"❌ Error processing image {image_path}: {e}")
                
        return len(images) > 0
        
    except Exception as e:
        print(f"❌ Error in image extraction: {e}")
        return False

async def test_caption_objects():
    """Test the caption_objects function"""
    print("\nTesting caption_objects function...")
    
    # Import the function from server.py
    from server import caption_objects
    
    # First extract some images
    from server import extract_images_from_pdf
    
    pdf_path = "test_maintenance_manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file {pdf_path} not found")
        return False
    
    try:
        images = extract_images_from_pdf(pdf_path)
        if not images:
            print("❌ No images extracted to test")
            return False
        
        # Test the first image
        test_image = images[0]
        print(f"Testing caption generation for: {test_image}")
        
        # This will test our fix for the OpenAI image format issue
        result = await caption_objects(test_image)
        
        print(f"✅ Caption generation result:")
        print(f"   Caption: {result.get('caption', 'No caption')}")
        print(f"   Objects: {result.get('objects', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in caption_objects test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Image Processing Fixes")
    print("=" * 50)
    
    # Test 1: Image extraction
    extraction_success = test_image_extraction()
    
    # Test 2: Caption objects (async)
    if extraction_success:
        asyncio.run(test_caption_objects())
    
    print("\n" + "=" * 50)
    print("✅ Image processing tests completed")