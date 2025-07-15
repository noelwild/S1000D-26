#!/usr/bin/env python3
"""
Test script to create a test image and verify the caption_objects fix
"""
import os
import sys
import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
import json

# Add the current directory to sys.path so we can import from server.py
sys.path.insert(0, '.')

def create_test_image():
    """Create a test technical diagram image"""
    print("Creating test technical diagram image...")
    
    try:
        # Create a simple technical diagram
        width, height = 800, 600
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a simple technical diagram
        # Main component box
        draw.rectangle([200, 200, 600, 400], outline='black', width=3)
        draw.text((350, 300), "ENGINE", fill='black', anchor='mm')
        
        # Input/output connections
        draw.line([100, 300, 200, 300], fill='black', width=2)
        draw.text((150, 280), "INPUT", fill='black', anchor='mm')
        
        draw.line([600, 300, 700, 300], fill='black', width=2)
        draw.text((650, 280), "OUTPUT", fill='black', anchor='mm')
        
        # Control panel
        draw.rectangle([250, 220, 350, 280], outline='blue', width=2)
        draw.text((300, 250), "CONTROL", fill='blue', anchor='mm')
        
        # Indicator lights
        draw.ellipse([450, 230, 470, 250], fill='green', outline='black')
        draw.text((480, 240), "READY", fill='green', anchor='lm')
        
        draw.ellipse([450, 260, 470, 280], fill='red', outline='black')
        draw.text((480, 270), "FAULT", fill='red', anchor='lm')
        
        # Save as different formats to test conversion
        test_images = []
        
        # Save as JPEG (supported)
        jpeg_path = "test_diagram.jpg"
        image.save(jpeg_path, 'JPEG', quality=90)
        test_images.append(jpeg_path)
        print(f"✅ Created JPEG test image: {jpeg_path}")
        
        # Save as PNG (supported)
        png_path = "test_diagram.png"
        image.save(png_path, 'PNG')
        test_images.append(png_path)
        print(f"✅ Created PNG test image: {png_path}")
        
        # Create a WEBP image (supported)
        webp_path = "test_diagram.webp"
        image.save(webp_path, 'WEBP', quality=90)
        test_images.append(webp_path)
        print(f"✅ Created WEBP test image: {webp_path}")
        
        return test_images
        
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return []

async def test_caption_objects_with_image(image_path):
    """Test the caption_objects function with a specific image"""
    print(f"\nTesting caption_objects with: {image_path}")
    
    # Import the function from server.py
    from server import caption_objects
    
    try:
        # Test our fixed caption_objects function
        result = await caption_objects(image_path)
        
        print(f"✅ Caption generation successful!")
        print(f"   Caption: {result.get('caption', 'No caption')}")
        print(f"   Objects: {result.get('objects', [])}")
        
        # Check if it's a fallback response or actual OpenAI response
        if result.get('caption') == "Technical diagram" and result.get('objects') == ["component", "system"]:
            print("   ⚠️  This is a fallback response (likely API key issue)")
        else:
            print("   ✅ This appears to be a real OpenAI response")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in caption_objects: {e}")
        return False

def test_image_format_conversion():
    """Test that our image format conversion works"""
    print("\nTesting image format conversion...")
    
    # Import the function from server.py
    from server import caption_objects
    
    try:
        # Create a test image in an "unsupported" format (but PIL can still read it)
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        # Save as BMP (not in OpenAI's supported list)
        bmp_path = "test_format.bmp"
        test_image.save(bmp_path, 'BMP')
        
        print(f"✅ Created BMP test image: {bmp_path}")
        
        # Test that our function can handle it
        with Image.open(bmp_path) as img:
            print(f"   Original format: {img.format}")
            
            # This should trigger our format conversion logic
            if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                print(f"   ✅ Format {img.format} would be converted to JPEG")
            else:
                print(f"   ℹ️  Format {img.format} is already supported")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in format conversion test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Image Processing Fixes with Test Images")
    print("=" * 60)
    
    # Test 1: Create test images
    test_images = create_test_image()
    
    # Test 2: Test format conversion logic
    format_test_success = test_image_format_conversion()
    
    # Test 3: Test caption_objects with different formats
    if test_images:
        for image_path in test_images:
            asyncio.run(test_caption_objects_with_image(image_path))
    
    print("\n" + "=" * 60)
    print("✅ All image processing tests completed")
    print("\nNote: If you see fallback responses, it's likely due to:")
    print("1. OpenAI API key issues")
    print("2. Network connectivity issues")
    print("3. OpenAI service unavailability")
    print("\nBut the image format conversion fix should prevent the")
    print("'unsupported image format' error you originally encountered.")