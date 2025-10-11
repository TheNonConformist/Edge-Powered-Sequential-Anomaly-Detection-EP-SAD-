import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Import our components (they should be in the same folder)
from sequence_engine import SequenceLogicEngine

class EnhancedReIDModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Enhanced ReID using: {self.device}")
        
        # Use larger pre-trained model for better features
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
        self.model.eval()
        self.model.to(self.device)
        
        # Multiple transforms for robustness
        self.transforms = [
            transforms.Compose([
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((128, 64)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
    def extract_enhanced_features(self, image):
        """Extract multiple augmented features for robustness"""
        if image is None or image.size == 0:
            return None
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            all_features = []
            for transform in self.transforms:
                input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model(input_tensor)
                    features = features.squeeze().cpu().numpy()
                    
                    if np.linalg.norm(features) > 0:
                        features = features / np.linalg.norm(features)
                        all_features.append(features)
            
            # Average features from different augmentations
            if all_features:
                avg_features = np.mean(all_features, axis=0)
                if np.linalg.norm(avg_features) > 0:
                    return avg_features / np.linalg.norm(avg_features)
            
            return None
            
        except Exception as e:
            print(f"Enhanced feature extraction error: {e}")
            return None

class PerformanceMonitor:
    def __init__(self):
        self.reid_success_rate = []
        self.tracking_accuracy = []
        self.false_positives = 0
        self.false_negatives = 0
        
    def log_reid_attempt(self, success, similarity_score):
        self.reid_success_rate.append(1 if success else 0)
        
    def calculate_metrics(self):
        if self.reid_success_rate:
            success_rate = np.mean(self.reid_success_rate[-100:])  # Last 100 attempts
            print(f"📊 ReID Success Rate: {success_rate:.1%}")
            return success_rate
        return 0.0

class EnhancedTracker:
    def __init__(self, max_distance=50, max_frames_skipped=50):
        # ... existing code ...
        
        # Add facial recognition system
        self.face_system = FacialIdentitySystem()
        self.face_to_track_map = {}  # Map face IDs to track IDs
        self.track_to_face_map = {}  # Map track IDs to face IDs
        
    def update(self, detections, frame=None, class_names=None):
        # ... existing detection code ...
        
        # STAGE 0: Face recognition for people
        face_detections = []
        for i, det in enumerate(detections):
            class_id = det[5]
            class_name = class_names.get(class_id, '')
            
            if class_name == 'person' and frame is not None:
                face_id, confidence = self.face_system.recognize_face(frame, det[:4])
                if face_id:
                    face_detections.append((i, face_id, confidence))
                    print(f"👤 Face recognized: {face_id} (confidence: {1-confidence:.2f})")
        
        # ... rest of your tracking logic ...
        
        # When creating new tracks for people, use face ID if available
        for i, centroid in enumerate(new_centroids):
            if i not in used_detections:
                # Check if this detection has a face ID
                face_id = None
                for face_det in face_detections:
                    if face_det[0] == i:
                        face_id = face_det[1]
                        break
                
                track_id = self.next_object_id
                
                # If we have a face ID, use permanent mapping
                if face_id and face_id in self.face_to_track_map:
                    # Reuse existing track ID for this face
                    track_id = self.face_to_track_map[face_id]
                    # Update existing object
                    if track_id in self.objects:
                        obj_data = self.objects[track_id]
                        obj_data.update({
                            'centroid': centroid,
                            'bbox': detections[i][:4],
                            'confidence': detections[i][4],
                            'class_id': detections[i][5],
                            'frames_skipped': 0,
                            'found_current_frame': True,
                            'last_seen': time.time(),
                            'appearance_count': obj_data.get('appearance_count', 0) + 1
                        })
                        print(f"🔁 REUSED TRACK: Face {face_id} -> Track {track_id}")
                    else:
                        # Create new object with existing track ID
                        self.objects[track_id] = {
                            'centroid': centroid,
                            'bbox': detections[i][:4],
                            'confidence': detections[i][4],
                            'class_id': detections[i][5],
                            'frames_skipped': 0,
                            'found_current_frame': True,
                            'features': new_features[i] if i < len(new_features) else None,
                            'first_seen': time.time(),
                            'last_seen': time.time(),
                            'appearance_count': 1,
                            'face_id': face_id
                        }
                else:
                    # Create new track
                    self.objects[track_id] = {
                        'centroid': centroid,
                        'bbox': detections[i][:4],
                        'confidence': detections[i][4],
                        'class_id': detections[i][5],
                        'frames_skipped': 0,
                        'found_current_frame': True,
                        'features': new_features[i] if i < len(new_features) else None,
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'appearance_count': 1,
                        'face_id': face_id
                    }
                    
                    # Update face-track mapping
                    if face_id:
                        self.face_to_track_map[face_id] = track_id
                        self.track_to_face_map[track_id] = face_id
                        print(f"🔗 MAPPED: Face {face_id} -> Track {track_id}")
                    
                    self.next_object_id += 1
                    
    def __init__(self, max_distance=50, max_frames_skipped=50):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance
        self.max_frames_skipped = max_frames_skipped
        self.reid_model = EnhancedReIDModel()
        self.performance_monitor = PerformanceMonitor()
        
        # Adaptive thresholds
        self.initial_reid_threshold = 0.6
        self.min_reid_threshold = 0.4
        self.feature_update_alpha = 0.1
        
        # Feature cache for multiple appearances
        self.feature_cache = {}
        self.max_cache_size = 5
        self.temporal_history = {}
        
        # Class-specific parameters
        self.class_params = {
            'person': {'max_distance': 80, 'reid_threshold': 0.6},
            'backpack': {'max_distance': 40, 'reid_threshold': 0.7},
            'car': {'max_distance': 120, 'reid_threshold': 0.5},
            'bicycle': {'max_distance': 60, 'reid_threshold': 0.65},
        }
        
    def _calculate_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_class_specific_params(self, class_id, class_names):
        """Different parameters for different object types"""
        class_name = class_names.get(class_id, 'person')
        return self.class_params.get(class_name, self.class_params['person'])
    
    def _extract_appearance_features(self, frame, bbox):
        """Extract appearance features for re-identification"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        try:
            # Extract object region
            object_region = frame[y1:y2, x1:x2]
            if object_region.size == 0:
                return None
            
            # Extract enhanced features
            features = self.reid_model.extract_enhanced_features(object_region)
            return features
            
        except Exception as e:
            print(f"Appearance feature extraction error: {e}")
            return None
    
    def _calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
            
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    
    def _update_feature_cache(self, track_id, new_features):
        """Maintain a cache of recent features for each object"""
        if track_id not in self.feature_cache:
            self.feature_cache[track_id] = []
        
        self.feature_cache[track_id].append(new_features)
        
        # Keep only recent features
        if len(self.feature_cache[track_id]) > self.max_cache_size:
            self.feature_cache[track_id].pop(0)
    
    def _get_best_feature_match(self, query_features, track_id):
        """Compare against all cached features and return best match"""
        if track_id not in self.feature_cache:
            return 0.0
            
        best_similarity = 0.0
        for cached_features in self.feature_cache[track_id]:
            if cached_features is not None:
                similarity = self._calculate_similarity(query_features, cached_features)
                best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def _calculate_adaptive_threshold(self, frames_skipped):
        """Lower threshold for objects that have been gone longer"""
        base_threshold = self.initial_reid_threshold
        # Gradually lower threshold up to 30 frames skipped
        adaptive_reduction = min(0.3, frames_skipped * 0.01)
        return max(self.min_reid_threshold, base_threshold - adaptive_reduction)
    
    def _update_object_features(self, obj_data, new_features):
        """Incrementally update object features"""
        if obj_data['features'] is None:
            obj_data['features'] = new_features
        else:
            # Moving average update
            obj_data['features'] = (1 - self.feature_update_alpha) * obj_data['features'] + \
                                 self.feature_update_alpha * new_features
            # Renormalize
            norm = np.linalg.norm(obj_data['features'])
            if norm > 0:
                obj_data['features'] /= norm
    
    def _check_temporal_consistency(self, track_id, new_bbox, new_time):
        """Check if the object movement is physically possible"""
        if track_id not in self.temporal_history:
            return True
            
        history = self.temporal_history[track_id]
        if len(history) < 2:
            return True
        
        # Calculate maximum possible movement based on time difference
        last_seen_time = history[-1]['time']
        time_diff = new_time - last_seen_time
        
        # Assume maximum speed of 5 meters per second (converted to pixels)
        max_pixel_movement = time_diff * 100  # Adjust based on your camera setup
        
        last_centroid = history[-1]['centroid']
        new_centroid = self._calculate_centroid(new_bbox)
        distance_moved = self._calculate_distance(last_centroid, new_centroid)
        
        return distance_moved <= max_pixel_movement
    
    def optimize_detections(self, detections, frame_shape):
        """Filter and optimize detections for better tracking"""
        optimized_detections = []
        
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            
            # Filter by confidence
            if confidence < 0.3:  # Increased threshold
                continue
                
            # Filter by size (remove too small/tiny detections)
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_shape[0] * frame_shape[1]
            
            if bbox_area < 0.001 * frame_area:  # At least 0.1% of frame
                continue
                
            # Filter by aspect ratio (remove unrealistic detections)
            aspect_ratio = (y2 - y1) / (x2 - x1)
            if aspect_ratio > 8 or aspect_ratio < 0.125:  # Too tall/wide
                continue
                
            optimized_detections.append(det)
        
        return optimized_detections

    def update(self, detections, frame=None, class_names=None):
        if class_names is None:
            class_names = {}
            
        # Optimize detections first
        if frame is not None:
            detections = self.optimize_detections(detections, frame.shape)
            
        new_centroids = [self._calculate_centroid(det[:4]) for det in detections]
        new_features = []
        
        # Extract features for new detections if frame is provided
        if frame is not None:
            for det in detections:
                features = self._extract_appearance_features(frame, det[:4])
                new_features.append(features)
        else:
            new_features = [None] * len(detections)
        
        # Mark all existing objects as not found in this frame
        for obj_id in self.objects:
            self.objects[obj_id]['found_current_frame'] = False
        
        # If no existing objects, create new ones
        if len(self.objects) == 0:
            for i, (centroid, features) in enumerate(zip(new_centroids, new_features)):
                self.objects[self.next_object_id] = {
                    'centroid': centroid,
                    'bbox': detections[i][:4],
                    'confidence': detections[i][4],
                    'class_id': detections[i][5],
                    'frames_skipped': 0,
                    'found_current_frame': True,
                    'features': features,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'appearance_count': 1
                }
                # Initialize temporal history
                self.temporal_history[self.next_object_id] = [{
                    'time': time.time(),
                    'centroid': centroid,
                    'bbox': detections[i][:4]
                }]
                print(f"🆕 NEW: Object {self.next_object_id} created")
                self.next_object_id += 1
        else:
            used_detections = set()
            
            # STAGE 1: Spatial matching for objects seen recently
            for obj_id, obj_data in self.objects.items():
                if len(new_centroids) == 0:
                    break
                    
                # Skip objects that have been lost for too long for spatial matching
                if obj_data['frames_skipped'] > 5:
                    continue
                    
                # Get class-specific parameters
                class_params = self._get_class_specific_params(obj_data['class_id'], class_names)
                
                distances = [self._calculate_distance(obj_data['centroid'], centroid) 
                           for centroid in new_centroids]
                
                if distances:
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]
                    
                    # Spatial matching with relaxed threshold for recent objects
                    spatial_threshold = class_params['max_distance'] * (1 + obj_data['frames_skipped'] * 0.1)
                    
                    if min_distance < spatial_threshold and min_distance_idx not in used_detections:
                        # Check temporal consistency
                        if self._check_temporal_consistency(obj_id, detections[min_distance_idx][:4], time.time()):
                            obj_data['centroid'] = new_centroids[min_distance_idx]
                            obj_data['bbox'] = detections[min_distance_idx][:4]
                            obj_data['confidence'] = detections[min_distance_idx][4]
                            obj_data['class_id'] = detections[min_distance_idx][5]
                            obj_data['frames_skipped'] = 0
                            obj_data['found_current_frame'] = True
                            obj_data['last_seen'] = time.time()
                            
                            # Update features
                            if new_features[min_distance_idx] is not None:
                                self._update_object_features(obj_data, new_features[min_distance_idx])
                                self._update_feature_cache(obj_id, new_features[min_distance_idx])
                            
                            # Update temporal history
                            if obj_id not in self.temporal_history:
                                self.temporal_history[obj_id] = []
                            self.temporal_history[obj_id].append({
                                'time': time.time(),
                                'centroid': new_centroids[min_distance_idx],
                                'bbox': detections[min_distance_idx][:4]
                            })
                            # Keep only recent history
                            if len(self.temporal_history[obj_id]) > 10:
                                self.temporal_history[obj_id].pop(0)
                            
                            used_detections.add(min_distance_idx)
                            print(f"📍 SPATIAL MATCH: Object {obj_id} tracked")
            
            # STAGE 2: Appearance-based re-identification for remaining detections
            if frame is not None:
                for i, (centroid, features) in enumerate(zip(new_centroids, new_features)):
                    if i not in used_detections and features is not None:
                        best_match_id = None
                        best_similarity = 0.0
                        
                        for obj_id, obj_data in self.objects.items():
                            if not obj_data['found_current_frame']:
                                # Use adaptive threshold based on how long object has been missing
                                current_threshold = self._calculate_adaptive_threshold(
                                    obj_data['frames_skipped']
                                )
                                
                                # Check against cached features
                                similarity = self._get_best_feature_match(features, obj_id)
                                
                                if similarity > best_similarity and similarity > current_threshold:
                                    best_similarity = similarity
                                    best_match_id = obj_id
                        
                        if best_match_id is not None:
                            # Re-identify object
                            obj_data = self.objects[best_match_id]
                            obj_data['centroid'] = centroid
                            obj_data['bbox'] = detections[i][:4]
                            obj_data['confidence'] = detections[i][4]
                            obj_data['class_id'] = detections[i][5]
                            obj_data['frames_skipped'] = 0
                            obj_data['found_current_frame'] = True
                            obj_data['last_seen'] = time.time()
                            obj_data['appearance_count'] += 1
                            
                            # Update features incrementally
                            self._update_object_features(obj_data, features)
                            self._update_feature_cache(best_match_id, features)
                            
                            # Update temporal history
                            if best_match_id not in self.temporal_history:
                                self.temporal_history[best_match_id] = []
                            self.temporal_history[best_match_id].append({
                                'time': time.time(),
                                'centroid': centroid,
                                'bbox': detections[i][:4]
                            })
                            if len(self.temporal_history[best_match_id]) > 10:
                                self.temporal_history[best_match_id].pop(0)
                            
                            used_detections.add(i)
                            self.performance_monitor.log_reid_attempt(True, best_similarity)
                            print(f"🔁 RE-ID: Object {best_match_id} (similarity: {best_similarity:.3f}, appearances: {obj_data['appearance_count']}, skipped: {obj_data['frames_skipped']} frames)")
                        else:
                            self.performance_monitor.log_reid_attempt(False, best_similarity)
            
            # STAGE 3: Create new objects for remaining detections
            for i, centroid in enumerate(new_centroids):
                if i not in used_detections:
                    self.objects[self.next_object_id] = {
                        'centroid': centroid,
                        'bbox': detections[i][:4],
                        'confidence': detections[i][4],
                        'class_id': detections[i][5],
                        'frames_skipped': 0,
                        'found_current_frame': True,
                        'features': new_features[i] if i < len(new_features) else None,
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'appearance_count': 1
                    }
                    # Initialize temporal history
                    self.temporal_history[self.next_object_id] = [{
                        'time': time.time(),
                        'centroid': centroid,
                        'bbox': detections[i][:4]
                    }]
                    print(f"🆕 NEW: Object {self.next_object_id} created (unmatched detection)")
                    self.next_object_id += 1
        
        # Update frames_skipped for objects not found
        objects_to_remove = []
        for obj_id, obj_data in self.objects.items():
            if not obj_data['found_current_frame']:
                obj_data['frames_skipped'] += 1
                
                # Remove object if skipped too many frames
                if obj_data['frames_skipped'] > self.max_frames_skipped:
                    objects_to_remove.append(obj_id)
        
        # Remove old objects
        for obj_id in objects_to_remove:
            appearance_count = self.objects[obj_id]['appearance_count']
            print(f"🗑️ REMOVED: Object {obj_id} (lost for {self.objects[obj_id]['frames_skipped']} frames, total appearances: {appearance_count})")
            del self.objects[obj_id]
            if obj_id in self.feature_cache:
                del self.feature_cache[obj_id]
            if obj_id in self.temporal_history:
                del self.temporal_history[obj_id]
        
        # Prepare tracks output
        tracks = []
        for obj_id, obj_data in self.objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            tracks.append({
                'track_id': obj_id,
                'bbox': [x1, y1, x2, y2],
                'centroid': obj_data['centroid'],
                'confidence': obj_data['confidence'],
                'class_id': obj_data['class_id'],
                'first_seen': obj_data.get('first_seen', time.time()),
                'last_seen': obj_data.get('last_seen', time.time()),
                'appearance_count': obj_data.get('appearance_count', 1)
            })
        
        # Calculate performance metrics every 100 frames
        if self.next_object_id % 100 == 0:
            self.performance_monitor.calculate_metrics()
        
        return tracks

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
    Complete intelligent system with fine-tuned ReID
    """
    def __init__(self, model_name='yolov8n.pt'):
        print("🧠 Initializing EP-SAD Intelligent Surveillance System...")
        
        # Core components
        self.detector = YOLO(model_name)
        self.class_names = self.detector.names
        self.tracker = EnhancedTracker(max_distance=100, max_frames_skipped=50)
        
        # Analysis components
        self.zone_analyzer = ZoneAnalyzer()
        self.sequence_engine = SequenceLogicEngine()
        
        # Visualization
        self.colors = {}
        self.tracking_history = {}
        self.velocity_history = {}
        
        # Statistics
        self.frame_count = 0
        self.total_alerts = 0
        self.max_alerts = 4
        self.alert_history = []
        self.alerted_objects = set()
        
        # Setup system
        self._setup_system()
        
        print("✓ Intelligent System Ready!")
        print("✓ Fine-Tuned ReID Tracking Active!")
        print("✓ Enhanced Feature Extraction Enabled!")

    def _setup_tracking_parameters(self):
        """Fine-tuned parameters based on use case"""
        self.tracking_params = {
            'indoor': {
                'max_distance': 60,
                'max_frames_skipped': 75,
                'reid_threshold': 0.65,
                'feature_update_alpha': 0.05
            },
            'outdoor': {
                'max_distance': 100,
                'max_frames_skipped': 30,
                'reid_threshold': 0.55,
                'feature_update_alpha': 0.1
            }
        }
        # Default to outdoor settings
        self.environment = 'outdoor'

    def _generate_unique_alerts_per_person(self, tracks, object_properties):
        """Generate alerts with ReID tracking"""
        alerts = []
        
        if not tracks:
            return alerts
        
        # Priority system for alerts
        alert_priority = {
            'unattended_bag': 100,
            'restricted_zone': 90, 
            'loitering': 80,
            'object_left_behind': 85,
            'suspicious_activity': 95,
            'object_detected': 10
        }
        
        # Get sequence engine alerts
        sequence_alerts = []
        for track_id, properties in object_properties.items():
            if track_id not in self.alerted_objects:
                object_alerts = self.sequence_engine.update_object_state(
                    track_id, 
                    properties['class_name'], 
                    properties
                )
                sequence_alerts.extend(object_alerts)
        
        # Filter sequence alerts
        unique_alerts = []
        alerted_tracks = set()
        
        for alert in sequence_alerts:
            track_id = alert['object_id']
            if track_id not in alerted_tracks and track_id not in self.alerted_objects:
                unique_alerts.append(alert)
                alerted_tracks.add(track_id)
                self.alerted_objects.add(track_id)
        
        # Use highest priority sequence alert
        if unique_alerts:
            unique_alerts.sort(key=lambda x: alert_priority.get(x.get('rule', 'object_detected'), 0), reverse=True)
            highest_alert = unique_alerts[0]
            alerts.append(highest_alert)
        else:
            # Create detection alerts
            for track in tracks:
                track_id = track['track_id']
                class_name = self.class_names[track['class_id']]
                appearance_count = track.get('appearance_count', 1)
                
                if track_id not in self.alerted_objects:
                    if appearance_count == 1:
                        alert = {
                            'rule': 'object_detected',
                            'severity': 'LOW',
                            'message': f'New {class_name} detected (ID: {track_id})',
                            'object_id': track_id,
                            'location': 'Camera Feed',
                            'confidence': track['confidence']
                        }
                    else:
                        alert = {
                            'rule': 'object_detected',
                            'severity': 'LOW', 
                            'message': f'{class_name} reappeared (ID: {track_id}) - {appearance_count} appearances',
                            'object_id': track_id,
                            'location': 'Camera Feed',
                            'confidence': track['confidence']
                        }
                    
                    alerts.append(alert)
                    self.alerted_objects.add(track_id)
                    break
        
        return alerts[:1]

    def _setup_system(self):
        self.demo_zones_setup = False
        self._setup_tracking_parameters()
    
    def setup_zones_based_on_resolution(self, width, height):
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
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def calculate_velocity(self, track_id, current_centroid):
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []
        
        history = self.velocity_history[track_id]
        history.append({'time': time.time(), 'position': current_centroid})
        
        # Keep only recent history
        history = [h for h in history if time.time() - h['time'] < 2]
        self.velocity_history[track_id] = history
        
        if len(history) >= 2:
            total_distance = 0
            for i in range(1, len(history)):
                dist = np.sqrt((history[i]['position'][0] - history[i-1]['position'][0])**2 +
                              (history[i]['position'][1] - history[i-1]['position'][1])**2)
                total_distance += dist
            
            time_span = history[-1]['time'] - history[0]['time']
            if time_span > 0:
                velocity = total_distance / time_span
                return velocity, total_distance < 20
        
        return 0, False
    
    def analyze_frame(self, frame):
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
                    
                    if class_name in ['person', 'backpack', 'handbag', 'suitcase', 'bag', 'car', 'bicycle', 'motorcycle']:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Update tracker with frame for feature extraction
        tracks = self.tracker.update(detections, frame, self.class_names)
        
        # Prepare data for sequence engine
        object_properties = {}
        
        for track in tracks:
            track_id = track['track_id']
            centroid = track['centroid']
            class_name = self.class_names[track['class_id']]
            
            velocity, is_stationary = self.calculate_velocity(track_id, centroid)
            
            zones_inside, zone_names = self.zone_analyzer.check_object_in_zones(
                track_id, centroid, track['bbox']
            )
            
            nearby = []
            for other_track in tracks:
                if other_track['track_id'] != track_id:
                    distance = np.sqrt((centroid[0] - other_track['centroid'][0])**2 +
                                      (centroid[1] - other_track['centroid'][1])**2)
                    if distance < 150:
                        nearby.append(other_track['track_id'])
            
            object_properties[track_id] = {
                'position': centroid,
                'velocity': velocity,
                'stationary': is_stationary,
                'zone': zone_names[0] if zone_names else None,
                'nearby_objects': nearby,
                'class_name': class_name,
                'first_seen': track.get('first_seen', time.time()),
                'last_seen': track.get('last_seen', time.time()),
                'appearance_count': track.get('appearance_count', 1)
            }
        
        # Generate alerts
        all_alerts = self._generate_unique_alerts_per_person(tracks, object_properties)
        
        # Manage alert history
        if len(self.alert_history) >= self.max_alerts:
            oldest_alert = self.alert_history[0]
            oldest_track_id = oldest_alert['object_id']
            if oldest_track_id in self.alerted_objects:
                self.alerted_objects.remove(oldest_track_id)
            self.alert_history.pop(0)
        
        if all_alerts:
            self.alert_history.append(all_alerts[0])
            self.total_alerts += 1
        
        if len(self.alert_history) > self.max_alerts:
            self.alert_history = self.alert_history[-self.max_alerts:]
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, tracks, all_alerts, object_properties)
        
        return annotated_frame, {
            'tracks': tracks,
            'alerts': all_alerts,
            'object_properties': object_properties
        }
    
    def _annotate_frame(self, frame, tracks, alerts, object_properties):
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
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Object label with appearance count
            props = object_properties.get(track_id, {})
            stationary_text = " (STATIONARY)" if props.get('stationary') else ""
            zone_text = f" in {props.get('zone')}" if props.get('zone') else ""
            appearance_count = props.get('appearance_count', 1)
            appearance_text = f" - #{appearance_count}" if appearance_count > 1 else ""
            
            label = f"ID:{track_id}{appearance_text} {class_name}{stationary_text}{zone_text}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Update tracking history
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = []
            self.tracking_history[track_id].append(centroid)
            
            # Draw path
            history = self.tracking_history[track_id][-10:]
            for i in range(1, len(history)):
                cv2.line(annotated_frame, history[i-1], history[i], color, 1)
        
        # Draw alerts
        for alert in alerts:
            track_id = alert['object_id']
            if track_id in object_properties:
                position = object_properties[track_id]['position']
                color = (0, 0, 255) if alert['severity'] == 'HIGH' else (0, 165, 255)
                cv2.circle(annotated_frame, position, 15, color, 2)
                cv2.putText(annotated_frame, "ALERT!", (position[0]+20, position[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw compact status panel
        self._draw_compact_panel(annotated_frame, alerts, width, height)
        
        return annotated_frame
    
    def _draw_compact_panel(self, frame, alerts, width, height):
        panel_width = 320
        panel_height = 140
        
        panel_x = width - panel_width - 10
        panel_y = 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 1)
        
        status_lines = [
            "EP-SAD FINE-TUNED ReID",
            f"Frame: {self.frame_count}",
            f"Alerts: {len(self.alert_history)}/{self.max_alerts}",
            f"Active Tracks: {len(self.tracking_history)}",
        ]
        
        if alerts:
            alert = alerts[0]
            short_msg = alert['message'][:25] + "..." if len(alert['message']) > 25 else alert['message']
            status_lines.append(f"Current: {short_msg}")
        
        for i, line in enumerate(status_lines):
            color = (255, 255, 255) if i > 0 else (0, 255, 255)
            font_size = 0.35 if i > 3 else 0.4
            thickness = 1
            y_pos = panel_y + 20 + i * 15
            cv2.putText(frame, line, (panel_x + 5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        cv2.putText(frame, "q:quit c:clear alerts", 
                   (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
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
    print("\n🎯 FINE-TUNED ReID TRACKING ACTIVE!")
    print("   - Enhanced CNN features with ResNet50")
    print("   - 3-stage matching with temporal consistency")
    print("   - Class-specific parameters")
    print("   - Feature caching and incremental updates")
    print("   - Performance monitoring")
    print("-" * 60)
    
    window_name = 'EP-SAD Fine-Tuned ReID Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            break
        
        annotated_frame, analysis_results = system.analyze_frame(frame)
        
        cv2.imshow(window_name, annotated_frame)
        
        for alert in analysis_results['alerts']:
            print(f"🚨 {alert['message']}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            system.alert_history.clear()
            system.alerted_objects.clear()
            system.total_alerts = 0
            print("🧹 All alerts cleared! Objects can be re-detected.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Tracking Summary:")
    print(f"   Frames analyzed: {system.frame_count}")
    print(f"   Total alerts generated: {system.total_alerts}")
    print(f"   Unique objects tracked: {len(system.tracking_history)}")

if __name__ == "__main__":
    main()