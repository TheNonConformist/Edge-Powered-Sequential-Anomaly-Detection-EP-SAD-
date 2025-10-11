import cv2
import sys
import os

def main():
    """
    Basic Video I/O script for EP-SAD project
    Reads a video file and displays it in a window
    """
    
    # Get the video file path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Using video file: {video_path}")
    else:
        video_path = 0  # 0 = default webcam
        print("No video file provided. Using webcam...")
    
    # Create VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        print("Make sure:")
        print("1. The video file exists and path is correct")
        print("2. Webcam is connected and not being used by another app")
        return
    
    print("✓ Video source opened successfully!")
    print("✓ Controls: Press 'q' to quit the video window")
    print("Loading video...")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Resolution: {width}x{height}")
    
    frame_count = 0
    
    # Main video processing loop
    while True:
        # Read next frame
        ret, frame = cap.read()
        frame_count += 1
        
        # Check if frame was successfully read
        if not ret:
            print("End of video or failed to read frame")
            break
        
        # Display frame number on the frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('EP-SAD Video Reader', frame)
        
        # Handle keyboard input - wait for 25ms
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Video playback ended. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()