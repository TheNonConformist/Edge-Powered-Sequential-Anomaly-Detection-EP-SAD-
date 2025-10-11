import cv2
import sys
import numpy as np
from ultralytics import YOLO
from sort_tracker import SORT

class DetectionTracker:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize YOLOv8 detector and SORT tracker
        """
        print("🚀 Initializing EP-SAD Detection & Tracking System...")
        
        # Initialize YOLOv8 detector
        print("Loading YOLOv8 model...")
        self.detector = YOLO(model_name)
        print(f"✓ YOLOv8 model loaded - can detect {len(self.detector.names)} object types")
        
        # Initialize SORT tracker
        self.tracker = SORT(max_age=20, min_hits=3, iou_threshold=0.3)
        print("✓ SORT tracker initialized")
        
        # Colors for different track IDs
        self.colors = {}
        
        # Tracking history
        self.tracking_history = {}
        
    def get_color(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.colors:
            # Generate a color based on track ID
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def detect_and_track(self, frame):
        """
        Run detection and tracking on a frame
        Returns: annotated frame, detections, tracks
        """
        # Run YOLOv8 detection
        results = self.detector(frame, verbose=False)
        
        # Format detections for SORT: [x1, y1, x2, y2, confidence]
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only track people and relevant objects (optional filter)
                    class_name = self.detector.names[class_id]
                    if class_name in ['person', 'backpack', 'handbag', 'suitcase']:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Convert to numpy array for SORT
        if detections:
            detections_array = np.array(detections)
        else:
            detections_array = np.empty((0, 6))
        
        # Update tracker with detections
        if len(detections_array) > 0:
            tracks = self.tracker.update(detections_array[:, :5])  # Only pass [x1,y1,x2,y2,score]
        else:
            tracks = self.tracker.update(np.empty((0, 5)))
        
        # Annotate frame with detections and tracks
        annotated_frame = frame.copy()
        
        # Draw detection bounding boxes
        for det in detections_array:
            x1, y1, x2, y2, conf, class_id = det
            class_name = self.detector.names[int(class_id)]
            
            # Draw detection box (semi-transparent)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.1, annotated_frame, 0.9, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Detection label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw tracking information
        tracked_objects = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Get color for this track
            color = self.get_color(track_id)
            
            # Draw track box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Track ID label
            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1)-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Store tracking info
            tracked_objects.append({
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [int((x1+x2)/2), int((y1+y2)/2)]
            })
            
            # Update tracking history
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = []
            self.tracking_history[track_id].append([int((x1+x2)/2), int((y1+y2)/2)])
            
            # Draw tracking path (last 20 points)
            history = self.tracking_history[track_id][-20:]
            for i in range(1, len(history)):
                cv2.line(annotated_frame, tuple(history[i-1]), tuple(history[i]), color, 2)
        
        return annotated_frame, detections_array, tracked_objects

def main():
    """
    Main function for detection and tracking
    """
    # Get video source
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 0  # Webcam
    
    # Initialize detection tracker
    system = DetectionTracker()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {width}x{height} at {fps:.1f} FPS")
    print("\n🎯 EP-SAD Detection & Tracking Active!")
    print("   - Green boxes: Detections")
    print("   - Colored boxes: Tracking with IDs")
    print("   - Colored lines: Movement paths")
    print("   - Press 'q' to quit")
    print("   - Press 'c' to clear tracking history")
    
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("End of video stream")
            break
        
        # Run detection and tracking
        annotated_frame, detections, tracks = system.detect_and_track(frame)
        
        # Update counters
        total_detections += len(detections)
        total_tracks += len(tracks)
        
        # Display info on frame
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Tracks: {len(tracks)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('EP-SAD Detection & Tracking', annotated_frame)
        
        # Print tracking info (first frame and when new tracks appear)
        if frame_count == 1 or len(tracks) > 0:
            print(f"\nFrame {frame_count}: {len(detections)} detections, {len(tracks)} tracks")
            for track in tracks:
                print(f"  📍 Track ID: {track['track_id']} at {track['center']}")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            system.tracking_history.clear()
            print("🗑️ Tracking history cleared!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 EP-SAD Tracking Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Total detections: {total_detections}")
    print(f"   Total tracks: {total_tracks}")
    print(f"   Unique objects tracked: {len(system.tracking_history)}")

if __name__ == "__main__":
    main()