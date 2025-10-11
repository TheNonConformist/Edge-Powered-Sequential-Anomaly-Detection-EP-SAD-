import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO

# Import our components (they should be in the same folder)
from sequence_engine import SequenceLogicEngine

# Reuse our existing components (copy the classes here or import them)
class SimpleTracker:
    # ... [COPY THE EXACT SimpleTracker CLASS FROM BEFORE] ...
    def __init__(self, max_distance=50, max_frames_skipped=10):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance
        self.max_frames_skipped = max_frames_skipped
        
    def _calculate_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections):
        new_centroids = [self._calculate_centroid(det[:4]) for det in detections]
        
        if len(self.objects) == 0:
            for centroid in new_centroids:
                self.objects[self.next_object_id] = {
                    'centroid': centroid,
                    'bbox': detections[new_centroids.index(centroid)][:4],
                    'confidence': detections[new_centroids.index(centroid)][4],
                    'class_id': detections[new_centroids.index(centroid)][5],
                    'frames_skipped': 0
                }
                self.next_object_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
            
            distance_matrix = np.zeros((len(object_centroids), len(new_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, new_centroid in enumerate(new_centroids):
                    distance_matrix[i, j] = self._calculate_distance(obj_centroid, new_centroid)
            
            used_detections = set()
            for i, obj_id in enumerate(object_ids):
                if len(new_centroids) == 0:
                    break
                    
                min_distance_idx = np.argmin(distance_matrix[i])
                min_distance = distance_matrix[i, min_distance_idx]
                
                if min_distance < self.max_distance and min_distance_idx not in used_detections:
                    self.objects[obj_id]['centroid'] = new_centroids[min_distance_idx]
                    self.objects[obj_id]['bbox'] = detections[min_distance_idx][:4]
                    self.objects[obj_id]['confidence'] = detections[min_distance_idx][4]
                    self.objects[obj_id]['class_id'] = detections[min_distance_idx][5]
                    self.objects[obj_id]['frames_skipped'] = 0
                    used_detections.add(min_distance_idx)
                else:
                    self.objects[obj_id]['frames_skipped'] += 1
            
            for j, centroid in enumerate(new_centroids):
                if j not in used_detections:
                    self.objects[self.next_object_id] = {
                        'centroid': centroid,
                        'bbox': detections[j][:4],
                        'confidence': detections[j][4],
                        'class_id': detections[j][5],
                        'frames_skipped': 0
                    }
                    self.next_object_id += 1
        
        objects_to_remove = []
        for obj_id, obj_data in self.objects.items():
            if obj_data['frames_skipped'] > self.max_frames_skipped:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.objects[obj_id]
        
        tracks = []
        for obj_id, obj_data in self.objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            tracks.append({
                'track_id': obj_id,
                'bbox': [x1, y1, x2, y2],
                'centroid': obj_data['centroid'],
                'confidence': obj_data['confidence'],
                'class_id': obj_data['class_id']
            })
        
        return tracks

class ZoneAnalyzer:
    # ... [COPY THE EXACT ZoneAnalyzer CLASS FROM BEFORE] ...
    def __init__(self):
        self.zones = {}
        self.zone_colors = {}
        self.next_zone_id = 0
        
    def add_rectangle_zone(self, name, x1, y1, x2, y2, color=None):
        zone_id = self.next_zone_id
        self.zones[zone_id] = {
            'name': name,
            'type': 'rectangle',
            'coords': [x1, y1, x2, y2],
            'objects_inside': set()
        }
        
        if color is None:
            hue = (zone_id * 60) % 180
            self.zone_colors[zone_id] = (hue, 255, 255)
        else:
            self.zone_colors[zone_id] = color
            
        self.next_zone_id += 1
        return zone_id
    
    def is_point_in_rectangle(self, point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def check_object_in_zones(self, track_id, centroid, bbox):
        zones_inside = []
        zone_names = []
        
        for zone_id, zone in self.zones.items():
            if zone['type'] == 'rectangle':
                is_inside = self.is_point_in_rectangle(centroid, zone['coords'])
            else:
                is_inside = False
            
            if is_inside:
                zones_inside.append(zone_id)
                zone_names.append(zone['name'])
        
        return zones_inside, zone_names
    
    def draw_zones(self, frame):
        for zone_id, zone in self.zones.items():
            color_hsv = self.zone_colors[zone_id]
            color_bgr = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
            color_bgr = tuple(map(int, color_bgr))
            
            if zone['type'] == 'rectangle':
                x1, y1, x2, y2 = zone['coords']
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(frame, zone['name'], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        return frame

class IntelligentSurveillanceSystem:
    """
    Complete intelligent system with sequence logic engine
    """
    def __init__(self, model_name='yolov8n.pt'):
        print("🧠 Initializing EP-SAD Intelligent Surveillance System...")
        
        # Core components
        self.detector = YOLO(model_name)
        self.class_names = self.detector.names
        self.tracker = SimpleTracker(max_distance=100, max_frames_skipped=30)
        
        # Analysis components
        self.zone_analyzer = ZoneAnalyzer()
        self.sequence_engine = SequenceLogicEngine()
        
        # Visualization
        self.colors = {}
        self.tracking_history = {}
        self.velocity_history = {}  # Track object movement for stationary detection
        
        # Statistics
        self.frame_count = 0
        self.total_alerts = 0
        
        # Setup system
        self._setup_system()
        
        print("✓ Intelligent System Ready!")
        print("✓ Sequence Logic Engine Active!")
        print("✓ Monitoring complex behavioral patterns")
    
    def _setup_system(self):
        """Setup zones and initial configuration"""
        self.demo_zones_setup = False
    
    def setup_zones_based_on_resolution(self, width, height):
        """Setup monitoring zones"""
        if self.demo_zones_setup:
            return
            
        # Restricted area
        self.zone_analyzer.add_rectangle_zone(
            "RESTRICTED AREA", 
            width - 300, 50, width - 50, 200,
            color=(0, 255, 255)
        )
        
        # Drop-off point
        self.zone_analyzer.add_rectangle_zone(
            "DROP-OFF POINT",
            width//2 - 150, height - 200, width//2 + 150, height - 50,
            color=(30, 255, 255)
        )
        
        self.demo_zones_setup = True
        print("✓ Monitoring zones configured")
    
    def get_color(self, track_id):
        """Get consistent color for track ID"""
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def calculate_velocity(self, track_id, current_centroid):
        """Calculate object velocity for stationary detection"""
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []
        
        history = self.velocity_history[track_id]
        history.append({'time': time.time(), 'position': current_centroid})
        
        # Keep only recent history (last 2 seconds)
        history = [h for h in history if time.time() - h['time'] < 2]
        self.velocity_history[track_id] = history
        
        if len(history) >= 2:
            # Calculate distance moved
            total_distance = 0
            for i in range(1, len(history)):
                dist = np.sqrt((history[i]['position'][0] - history[i-1]['position'][0])**2 +
                              (history[i]['position'][1] - history[i-1]['position'][1])**2)
                total_distance += dist
            
            time_span = history[-1]['time'] - history[0]['time']
            if time_span > 0:
                velocity = total_distance / time_span
                return velocity, total_distance < 20  # Stationary if moved less than 20 pixels in 2 seconds
        
        return 0, False
    
    def analyze_frame(self, frame):
        """Complete intelligent analysis of a frame"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        if not self.demo_zones_setup:
            self.setup_zones_based_on_resolution(width, height)
        
        # Run detection
        results = self.detector(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    if class_name in ['person', 'backpack', 'handbag', 'suitcase', 'bag']:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Prepare data for sequence engine
        object_properties = {}
        nearby_objects = {}
        
        # Calculate object relationships and properties
        for i, track in enumerate(tracks):
            track_id = track['track_id']
            centroid = track['centroid']
            class_name = self.class_names[track['class_id']]
            
            # Calculate velocity and stationary status
            velocity, is_stationary = self.calculate_velocity(track_id, centroid)
            
            # Check zone occupancy
            zones_inside, zone_names = self.zone_analyzer.check_object_in_zones(
                track_id, centroid, track['bbox']
            )
            
            # Find nearby objects (within 150 pixels)
            nearby = []
            for other_track in tracks:
                if other_track['track_id'] != track_id:
                    distance = np.sqrt((centroid[0] - other_track['centroid'][0])**2 +
                                      (centroid[1] - other_track['centroid'][1])**2)
                    if distance < 150:
                        nearby.append(other_track['track_id'])
            
            # Store properties for sequence engine
            object_properties[track_id] = {
                'position': centroid,
                'velocity': velocity,
                'stationary': is_stationary,
                'zone': zone_names[0] if zone_names else None,
                'nearby_objects': nearby,
                'class_name': class_name
            }
            
            nearby_objects[track_id] = nearby
        
        # Update sequence logic engine
        alerts = []
        for track_id, properties in object_properties.items():
            object_alerts = self.sequence_engine.update_object_state(
                track_id, 
                properties['class_name'], 
                properties
            )
            alerts.extend(object_alerts)
        
        # Update alert count
        self.total_alerts += len(alerts)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, tracks, alerts, object_properties)
        
        return annotated_frame, {
            'tracks': tracks,
            'alerts': alerts,
            'object_properties': object_properties
        }
    
    def _annotate_frame(self, frame, tracks, alerts, object_properties):
        """Annotate frame with intelligent analysis results"""
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw zones
        annotated_frame = self.zone_analyzer.draw_zones(annotated_frame)
        
        # Draw tracks and information
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            centroid = track['centroid']
            class_name = self.class_names[track['class_id']]
            
            color = self.get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Object label with additional info
            props = object_properties.get(track_id, {})
            stationary_text = " (STATIONARY)" if props.get('stationary') else ""
            zone_text = f" in {props.get('zone')}" if props.get('zone') else ""
            
            label = f"ID:{track_id} {class_name}{stationary_text}{zone_text}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update tracking history
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = []
            self.tracking_history[track_id].append(centroid)
            
            # Draw path
            history = self.tracking_history[track_id][-15:]
            for i in range(1, len(history)):
                cv2.line(annotated_frame, history[i-1], history[i], color, 2)
        
        # Draw alerts
        for alert in alerts:
            # Find object position for alert
            track_id = alert['object_id']
            if track_id in object_properties:
                position = object_properties[track_id]['position']
                
                # Draw alert indicator
                color = (0, 0, 255) if alert['severity'] == 'HIGH' else (0, 165, 255)
                cv2.circle(annotated_frame, position, 20, color, 3)
                cv2.putText(annotated_frame, "ALERT!", (position[0]+25, position[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw intelligent status panel
        self._draw_intelligent_panel(annotated_frame, alerts, width, height)
        
        return annotated_frame
    
    def _draw_intelligent_panel(self, frame, alerts, width, height):
        """Draw intelligent status panel"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get statistics
        rule_stats = self.sequence_engine.get_rule_statistics()
        active_alerts = self.sequence_engine.get_active_alerts()
        
        status_lines = [
            "EP-SAD INTELLIGENT SYSTEM",
            f"Frame: {self.frame_count}",
            f"Active Alerts: {len(active_alerts)}",
            f"Total Alerts: {self.total_alerts}",
            "",
            "DETECTION RULES:",
        ]
        
        # Add rule statistics
        for rule_name, stats in rule_stats.items():
            status_lines.append(f"  {rule_name}: {stats['recent_alerts']} recent")
        
        # Add current alerts
        if alerts:
            status_lines.extend(["", "CURRENT ALERTS:"])
            for alert in alerts[-3:]:  # Show last 3 alerts
                short_msg = alert['message'].split(':')[1] if ':' in alert['message'] else alert['message']
                status_lines.append(f"  {short_msg[:25]}...")
        
        # Draw status text
        for i, line in enumerate(status_lines):
            color = (255, 255, 255) if i > 0 else (0, 255, 255)
            font_size = 0.45 if i > 0 else 0.6
            thickness = 1 if i > 0 else 2
            y_pos = 40 + i * 20
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset stats", 
                   (width - 350, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """
    Main function for intelligent surveillance system
    """
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 0
    
    print("🚀 Starting EP-SAD Intelligent Surveillance System...")
    system = IntelligentSurveillanceSystem()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {width}x{height} at {fps:.1f} FPS")
    print("\n🎯 INTELLIGENT SURVEILLANCE ACTIVE!")
    print("   - Sequence Logic: Detecting complex event patterns")
    print("   - Rules: Unattended bags, Zone violations, Loitering")
    print("   - Red circles: High-severity alerts")
    print("   - Orange circles: Medium-severity alerts")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            break
        
        # Run intelligent analysis
        annotated_frame, analysis_results = system.analyze_frame(frame)
        
        # Display
        cv2.imshow('EP-SAD Intelligent Surveillance', annotated_frame)
        
        # Print new alerts to console
        for alert in analysis_results['alerts']:
            print(f"🚨 {alert['message']} (Object {alert['object_id']})")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            system.sequence_engine.triggered_alerts.clear()
            system.total_alerts = 0
            print("📊 Statistics reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Intelligent System Summary:")
    print(f"   Frames analyzed: {system.frame_count}")
    print(f"   Total alerts generated: {system.total_alerts}")
    print(f"   Unique objects tracked: {len(system.tracking_history)}")
    
    # Print final rule statistics
    stats = system.sequence_engine.get_rule_statistics()
    print("\n📊 Rule Performance:")
    for rule_name, rule_stats in stats.items():
        print(f"   {rule_name}: {rule_stats['total_alerts']} total alerts")

if __name__ == "__main__":
    main()