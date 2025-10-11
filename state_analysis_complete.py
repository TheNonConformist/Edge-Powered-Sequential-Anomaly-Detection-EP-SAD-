import cv2
import sys
import numpy as np
from ultralytics import YOLO

print("🚀 Loading EP-SAD State Analysis System...")

# =============================================================================
# SIMPLE TRACKER
# =============================================================================

class SimpleTracker:
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

# =============================================================================
# ZONE ANALYZER
# =============================================================================

class ZoneAnalyzer:
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
        print(f"✓ Zone '{name}' added at [{x1}, {y1}, {x2}, {y2}]")
        return zone_id
    
    def add_polygon_zone(self, name, points, color=None):
        zone_id = self.next_zone_id
        self.zones[zone_id] = {
            'name': name,
            'type': 'polygon', 
            'coords': points,
            'objects_inside': set()
        }
        
        if color is None:
            hue = (zone_id * 60) % 180
            self.zone_colors[zone_id] = (hue, 255, 255)
        else:
            self.zone_colors[zone_id] = color
            
        self.next_zone_id += 1
        print(f"✓ Polygonal zone '{name}' added with {len(points)} points")
        return zone_id
    
    def is_point_in_rectangle(self, point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def is_point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def check_object_in_zones(self, track_id, centroid, bbox):
        zones_inside = []
        
        for zone_id, zone in self.zones.items():
            was_inside = track_id in zone['objects_inside']
            
            if zone['type'] == 'rectangle':
                is_inside = self.is_point_in_rectangle(centroid, zone['coords'])
            else:
                is_inside = self.is_point_in_polygon(centroid, zone['coords'])
            
            if is_inside:
                zone['objects_inside'].add(track_id)
                if not was_inside:
                    print(f"🚨 ENTER ZONE: Track {track_id} entered '{zone['name']}'")
                zones_inside.append(zone_id)
            else:
                if was_inside:
                    zone['objects_inside'].discard(track_id)
                    print(f"🚨 LEFT ZONE: Track {track_id} left '{zone['name']}'")
        
        return zones_inside
    
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
                cv2.putText(frame, f"Objects: {len(zone['objects_inside'])}", (x1, y1-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
                
            else:
                points = np.array(zone['coords'], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [points], color_bgr)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [points], True, color_bgr, 2)
                if len(zone['coords']) > 0:
                    label_x, label_y = zone['coords'][0]
                    cv2.putText(frame, zone['name'], (label_x, label_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        return frame
    
    def get_zone_occupancy(self):
        occupancy = {}
        for zone_id, zone in self.zones.items():
            occupancy[zone['name']] = {
                'count': len(zone['objects_inside']),
                'object_ids': list(zone['objects_inside'])
            }
        return occupancy

# =============================================================================
# INTERACTION ANALYZER
# =============================================================================

class InteractionAnalyzer:
    def __init__(self, proximity_threshold=100):
        self.proximity_threshold = proximity_threshold
        self.interaction_history = {}
        self.object_states = {}
        
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def check_proximity(self, tracks, class_names):
        proximity_events = []
        
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks):
                if i >= j:
                    continue
                
                distance = self.calculate_distance(track1['centroid'], track2['centroid'])
                
                if distance < self.proximity_threshold:
                    event = self._analyze_interaction_type(track1, track2, distance, class_names)
                    proximity_events.append(event)
        
        return proximity_events
    
    def _analyze_interaction_type(self, track1, track2, distance, class_names):
        class1 = class_names[track1['class_id']]
        class2 = class_names[track2['class_id']]
        
        if ('person' in class1 and any(item in class2 for item in ['backpack', 'handbag', 'suitcase', 'bag'])) or \
           ('person' in class2 and any(item in class1 for item in ['backpack', 'handbag', 'suitcase', 'bag'])):
            
            if 'person' in class1:
                person_track, bag_track = track1, track2
                person_class, bag_class = class1, class2
            else:
                person_track, bag_track = track2, track1
                person_class, bag_class = class2, class1
            
            interaction_type = "CARRYING"
            description = f"Person {person_track['track_id']} carrying {bag_class} {bag_track['track_id']}"
            
        elif 'person' in class1 and 'person' in class2:
            interaction_type = "MEETING"
            description = f"Person {track1['track_id']} meeting Person {track2['track_id']}"
            
        else:
            interaction_type = "PROXIMITY"
            description = f"{class1} {track1['track_id']} near {class2} {track2['track_id']}"
        
        event = {
            'type': interaction_type,
            'description': description,
            'track1_id': track1['track_id'],
            'track2_id': track2['track_id'],
            'distance': distance,
            'position': track1['centroid']
        }
        
        interaction_key = (min(track1['track_id'], track2['track_id']), 
                          max(track1['track_id'], track2['track_id']))
        
        if interaction_key not in self.interaction_history:
            print(f"🤝 INTERACTION: {description} ({distance:.1f}px)")
            self.interaction_history[interaction_key] = event
        
        return event
    
    def check_object_left_behind(self, tracks, class_names):
        abandoned_events = []
        
        for track in tracks:
            class_name = class_names[track['class_id']]
            
            if any(item in class_name for item in ['backpack', 'handbag', 'suitcase', 'bag']):
                is_stationary = self._is_object_stationary(track['track_id'], track['centroid'])
                
                has_person_nearby = False
                for other_track in tracks:
                    if other_track['track_id'] == track['track_id']:
                        continue
                    
                    other_class = class_names[other_track['class_id']]
                    if 'person' in other_class:
                        distance = self.calculate_distance(track['centroid'], other_track['centroid'])
                        if distance < self.proximity_threshold * 1.5:
                            has_person_nearby = True
                            break
                
                if is_stationary and not has_person_nearby:
                    event_key = f"abandoned_{track['track_id']}"
                    if event_key not in self.object_states:
                        event = {
                            'type': 'ABANDONED_OBJECT',
                            'description': f"{class_name} {track['track_id']} may be left behind",
                            'track_id': track['track_id'],
                            'position': track['centroid']
                        }
                        abandoned_events.append(event)
                        self.object_states[event_key] = event
                        print(f"🚨 ABANDONMENT: {class_name} {track['track_id']} possibly left behind!")
        
        return abandoned_events
    
    def _is_object_stationary(self, track_id, current_position, movement_threshold=10):
        if track_id not in self.interaction_history:
            self.interaction_history[track_id] = {'positions': []}
        
        history = self.interaction_history[track_id]['positions']
        history.append(current_position)
        
        if len(history) > 10:
            history.pop(0)
        
        if len(history) >= 5:
            start_pos = history[0]
            end_pos = history[-1]
            distance_moved = self.calculate_distance(start_pos, end_pos)
            return distance_moved < movement_threshold
        
        return False
    
    def draw_interactions(self, frame, interactions, abandoned_objects):
        for interaction in interactions:
            pos = interaction['position']
            color = (0, 255, 255)
            
            cv2.circle(frame, pos, 8, color, -1)
            cv2.circle(frame, pos, 15, color, 2)
            
            label = f"{interaction['type']}"
            cv2.putText(frame, label, (pos[0] + 20, pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        for abandoned in abandoned_objects:
            pos = abandoned['position']
            color = (0, 0, 255)
            
            cv2.circle(frame, pos, 12, color, 3)
            cv2.putText(frame, "!", (pos[0]-5, pos[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            label = "LEFT BEHIND"
            cv2.putText(frame, label, (pos[0] + 20, pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

# =============================================================================
# MAIN STATE ANALYSIS SYSTEM
# =============================================================================

class StateAnalysisSystem:
    def __init__(self, model_name='yolov8n.pt'):
        print("🧠 Initializing EP-SAD State Analysis System...")
        
        self.detector = YOLO(model_name)
        self.class_names = self.detector.names
        self.tracker = SimpleTracker(max_distance=100, max_frames_skipped=30)
        
        self.zone_analyzer = ZoneAnalyzer()
        self.interaction_analyzer = InteractionAnalyzer(proximity_threshold=120)
        
        self.colors = {}
        self.tracking_history = {}
        
        self.demo_zones_setup = False
        
        print("✓ State Analysis System ready!")
        print("✓ Monitoring: Zone entries, Object interactions, Abandoned objects")
    
    def setup_zones_based_on_resolution(self, width, height):
        if self.demo_zones_setup:
            return
            
        # Restricted area (top-right)
        self.zone_analyzer.add_rectangle_zone(
            "RESTRICTED AREA", 
            width - 300, 50, width - 50, 200,
            color=(0, 255, 255)
        )
        
        # Drop-off point (bottom-center)
        self.zone_analyzer.add_rectangle_zone(
            "DROP-OFF POINT",
            width//2 - 150, height - 200, width//2 + 150, height - 50,
            color=(30, 255, 255)
        )
        
        # Meeting zone (polygon - center-left)
        meeting_points = [
            (100, height//2 - 100),
            (300, height//2 - 150), 
            (350, height//2 + 100),
            (150, height//2 + 150)
        ]
        self.zone_analyzer.add_polygon_zone(
            "MEETING ZONE",
            meeting_points,
            color=(120, 255, 255)
        )
        
        self.demo_zones_setup = True
        print("✓ Demo zones configured")
    
    def get_color(self, track_id):
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def analyze_frame(self, frame):
        height, width = frame.shape[:2]
        if not self.demo_zones_setup:
            self.setup_zones_based_on_resolution(width, height)
        
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
        
        tracks = self.tracker.update(detections)
        
        analysis_results = {
            'tracks': tracks,
            'zone_events': [],
            'interactions': [],
            'abandoned_objects': []
        }
        
        for track in tracks:
            centroid = track['centroid']
            zones_inside = self.zone_analyzer.check_object_in_zones(
                track['track_id'], centroid, track['bbox']
            )
            if zones_inside:
                analysis_results['zone_events'].append({
                    'track_id': track['track_id'],
                    'zones': zones_inside,
                    'position': centroid
                })
        
        analysis_results['interactions'] = self.interaction_analyzer.check_proximity(
            tracks, self.class_names
        )
        
        analysis_results['abandoned_objects'] = self.interaction_analyzer.check_object_left_behind(
            tracks, self.class_names
        )
        
        annotated_frame = self._annotate_frame(frame, analysis_results)
        
        return annotated_frame, analysis_results
    
    def _annotate_frame(self, frame, analysis_results):
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        annotated_frame = self.zone_analyzer.draw_zones(annotated_frame)
        
        for track in analysis_results['tracks']:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            centroid = track['centroid']
            class_name = self.class_names[track['class_id']]
            
            color = self.get_color(track_id)
            
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            label = f"ID:{track_id} {class_name}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = []
            self.tracking_history[track_id].append(centroid)
            
            history = self.tracking_history[track_id][-10:]
            for i in range(1, len(history)):
                cv2.line(annotated_frame, history[i-1], history[i], color, 2)
        
        annotated_frame = self.interaction_analyzer.draw_interactions(
            annotated_frame, 
            analysis_results['interactions'],
            analysis_results['abandoned_objects']
        )
        
        self._draw_status_panel(annotated_frame, analysis_results, width, height)
        
        return annotated_frame
    
    def _draw_status_panel(self, frame, analysis_results, width, height):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        status_lines = [
            "EP-SAD STATE ANALYSIS",
            f"Tracks: {len(analysis_results['tracks'])}",
            f"Interactions: {len(analysis_results['interactions'])}",
            f"Alerts: {len(analysis_results['abandoned_objects'])}",
            "",
            "ZONES:",
        ]
        
        occupancy = self.zone_analyzer.get_zone_occupancy()
        for zone_name, info in occupancy.items():
            status_lines.append(f"  {zone_name}: {info['count']} objects")
        
        for i, line in enumerate(status_lines):
            color = (255, 255, 255) if i > 0 else (0, 255, 255)
            font_size = 0.5 if i > 0 else 0.6
            thickness = 1 if i > 0 else 2
            cv2.putText(frame, line, (20, 40 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        cv2.putText(frame, "Press 'q' to quit | 'c' to clear", 
                   (width - 300, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 0
    
    system = StateAnalysisSystem()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {width}x{height} at {fps:.1f} FPS")
    print("\n🎯 EP-SAD State Analysis Active!")
    print("   - Colored boxes: Tracked objects")
    print("   - Colored zones: Restricted areas") 
    print("   - Yellow circles: Interactions")
    print("   - Red ! marks: Abandoned objects")
    print("   - Status panel: Live analysis")
    print("-" * 50)
    
    frame_count = 0
    alert_count = 0
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("End of video stream")
            break
        
        annotated_frame, analysis_results = system.analyze_frame(frame)
        
        alert_count += len(analysis_results['abandoned_objects'])
        
        cv2.imshow('EP-SAD State Analysis', annotated_frame)
        
        if analysis_results['abandoned_objects']:
            for alert in analysis_results['abandoned_objects']:
                print(f"🚨 ALERT: {alert['description']}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            system.tracking_history.clear()
            print("🗑️ Tracking history cleared!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 State Analysis Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Total alerts: {alert_count}")
    print(f"   Unique objects: {len(system.tracking_history)}")

if __name__ == "__main__":
    main()