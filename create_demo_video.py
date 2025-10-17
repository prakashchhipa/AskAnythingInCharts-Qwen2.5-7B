#!/usr/bin/env python3
"""
Create a video demo for GitHub README
Shows the HuggingFace Space in action
"""

import subprocess
import os
from pathlib import Path

def create_video_demo():
    """Create a video showing the demo in action"""
    
    print("🎬 Creating video demo...")
    print("This script will help you create a video of your HuggingFace Space in action.")
    print()
    print("Steps to create a video:")
    print("1. Open your HuggingFace Space: https://huggingface.co/spaces/prakashchhipa/chart-qa-demo-qwen2.5")
    print("2. Use a screen recording tool like:")
    print("   - OBS Studio (free)")
    print("   - QuickTime (Mac)")
    print("   - Windows Game Bar (Windows)")
    print("   - SimpleScreenRecorder (Linux)")
    print()
    print("3. Record a 30-60 second demo showing:")
    print("   - The interface loading")
    print("   - Selecting an example")
    print("   - Showing the comparison")
    print("   - Uploading a custom chart")
    print()
    print("4. Save the video as 'demo_video.mp4'")
    print("5. Add it to your README with:")
    print("   ![Video Demo](demo_video.mp4)")
    print()
    print("Alternative: Use Loom, LiceCap, or similar tools for quick GIF creation")

def create_screenshots():
    """Create screenshots for documentation"""
    
    print("📸 Creating screenshot guide...")
    print()
    print("Take these screenshots of your HuggingFace Space:")
    print("1. Main interface - showing the title and input areas")
    print("2. Example selection - showing the dropdown with examples")
    print("3. Results comparison - showing base vs fine-tuned answers")
    print("4. Custom upload - showing a custom chart being analyzed")
    print("5. Performance metrics - showing the accuracy improvements")
    print()
    print("Save them as:")
    print("- screenshot_interface.png")
    print("- screenshot_examples.png") 
    print("- screenshot_comparison.png")
    print("- screenshot_custom.png")
    print("- screenshot_metrics.png")

if __name__ == "__main__":
    print("🎥 Demo Video Creation Guide")
    print("=" * 40)
    print()
    
    create_video_demo()
    print()
    create_screenshots()
    print()
    print("✅ Use these guides to create engaging visual content for your README!")
