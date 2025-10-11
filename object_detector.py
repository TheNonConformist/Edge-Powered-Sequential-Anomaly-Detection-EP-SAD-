import cv2
import sys
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize YOLOv8 model for object detection
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_name)
        print(f"✓ Model '{model_name}' loaded successfully!")
        
        # Get class names
        self.class_names = self.model.names
        print(f"✓ Model can detect {len(self.class_names)} different object types")
        
    def detect_objects(self, frame):
        """
        Detect objects in a frame and return results
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)  # verbose=False to reduce output noise
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    })
        
        return detections, results[0].plot()  # Return both raw detections and annotated frame

def main():
    """
    Main function to run object detection on video/webcam
    """
    # Get video source from command line or use webcam
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 0  # Webcam
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video Properties: {width}x{height} at {fps:.1f} FPS")
    print("\n🎯 Object Detection Active!")
    print("   - Detecting: people, bags, vehicles, etc.")
    print("   - Press 'q' to quit")
    print("   - Press 'p' to pause/resume")
    
    paused = False
    frame_count = 0
    detection_count = 0
    
    while True:
        if not paused:
            # Read frame
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                print("End of video stream")
                break
            
            # Run object detection
            detections, annotated_frame = detector.detect_objects(frame)
            
            # Update detection counter
            detection_count += len(detections)
            
            # Display detection info on frame
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the frame with detections
            cv2.imshow('EP-SAD Object Detection', annotated_frame)
            
            # Print detection details in console (first frame and when objects detected)
            if frame_count == 1 or len(detections) > 0:
                print(f"\nFrame {frame_count}: Found {len(detections)} objects")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f} confidence")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Detection Summary:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total objects detected: {detection_count}")
    print(f"   Average detections per frame: {detection_count/frame_count:.2f}")

if __name__ == "__main__":
    main()