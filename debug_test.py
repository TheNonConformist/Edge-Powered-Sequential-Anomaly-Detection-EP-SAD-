import cv2
import sys
import numpy as np

print("🔍 DEBUG TEST: Starting basic OpenCV test...")

# Test 1: Basic Python and OpenCV
print("✓ Python and OpenCV imported successfully")

# Test 2: Try to open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam")
    print("Trying backup camera index 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("❌ ERROR: Cannot open any camera")
    print("Please check:")
    print("1. Webcam is connected")
    print("2. No other app is using the webcam")
    print("3. Webcam drivers are installed")
else:
    print("✓ Webcam opened successfully")
    
    # Test 3: Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read frame from webcam")
    else:
        print("✓ Frame read successfully")
        print(f"✓ Frame dimensions: {frame.shape}")
        
        # Test 4: Try to display frame
        cv2.imshow('Debug Test - Press Q to close', frame)
        print("✓ Displaying frame - you should see a window")
        print("✓ Press 'q' to close the window")
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("✓ Window closed successfully")

cap.release()
print("🎉 Debug test completed!")