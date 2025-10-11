from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import json
import threading
import time
from datetime import datetime
from ultralytics import YOLO

# Import our actual intelligent system
from intelligent_system import IntelligentSurveillanceSystem

# Shared state manager for communication between systems
class SharedStateManager:
    def __init__(self):
        self.alerts = []
        self.video_frame = None
        self.system_active = False
        self.stats = {
            'total_frames': 0,
            'total_alerts': 0,
            'active_tracks': 0,
            'current_fps': 0
        }
        self.last_update = datetime.now()
        self.intelligent_system = None
        self.lock = threading.Lock()
    
    def initialize_intelligent_system(self):
        """Initialize the intelligent system"""
        if self.intelligent_system is None:
            self.intelligent_system = IntelligentSurveillanceSystem()
            self.system_active = True
            print("✓ Intelligent system initialized in dashboard")
    
    def add_alert(self, alert_data):
        """Add a new alert to the dashboard"""
        with self.lock:
            alert = {
                'id': len(self.alerts) + 1,
                'timestamp': datetime.now().isoformat(),
                'type': alert_data.get('rule', 'Unknown'),
                'severity': alert_data.get('severity', 'medium'),
                'message': alert_data.get('message', ''),
                'object_id': alert_data.get('object_id', ''),
                'location': alert_data.get('location', ''),
                'acknowledged': False
            }
            self.alerts.insert(0, alert)  # Add to beginning for latest first
            self.stats['total_alerts'] += 1
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[:100]
    
    def update_frame(self, frame):
        """Update the current video frame"""
        with self.lock:
            self.video_frame = frame
            self.stats['total_frames'] += 1
    
    def get_recent_alerts(self, count=10):
        """Get most recent alerts"""
        with self.lock:
            return self.alerts[:count]
    
    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged"""
        with self.lock:
            for alert in self.alerts:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    return True
            return False

# Initialize Flask app and shared state manager
app = Flask(__name__)
shared_state = SharedStateManager()

def generate_intelligent_feed():
    """Generate video feed with actual intelligent system analysis"""
    # Initialize the intelligent system if not already done
    shared_state.initialize_intelligent_system()
    
    # Open video source
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera found, creating synthetic feed...")
        # Create a synthetic video feed if no camera available
        while True:
            # Create a synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some text
            cv2.putText(frame, "EP-SAD SECURITY DASHBOARD", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "LIVE FEED - DEMO MODE", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (50, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Simulate some moving objects
            for i in range(3):
                x = int(200 + 100 * np.sin(time.time() + i))
                y = int(200 + 100 * np.cos(time.time() + i))
                cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
            
            # Update shared state
            shared_state.update_frame(frame)
            
            # Apply aspect ratio handling to synthetic frame too
            height, width = frame.shape[:2]
            max_width, max_height = 1280, 720
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h, 1.0)
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    else:
        print("Camera found, using live intelligent feed...")
        while True:
            ret, frame = cap.read()
            if ret:
                # Run intelligent analysis on the frame
                try:
                    annotated_frame, analysis_results = shared_state.intelligent_system.analyze_frame(frame)
                    
                    # Process alerts from intelligent system
                    for alert in analysis_results['alerts']:
                        # Convert intelligent system alert to dashboard format
                        alert_data = {
                            'rule': alert.get('rule', 'unknown'),
                            'severity': alert.get('severity', 'medium').lower(),
                            'message': alert.get('message', ''),
                            'object_id': alert.get('object_id', ''),
                            'location': 'Camera Feed'
                        }
                        shared_state.add_alert(alert_data)
                    
                    # Update stats
                    shared_state.stats['active_tracks'] = len(analysis_results['tracks'])
                    
                    # Update shared state with annotated frame
                    shared_state.update_frame(annotated_frame)
                    
                    # Ensure proper aspect ratio and resize if needed
                    height, width = annotated_frame.shape[:2]
                    max_width, max_height = 1280, 720  # Standard HD resolution
                    
                    # Calculate scaling factor to maintain aspect ratio
                    scale_w = max_width / width
                    scale_h = max_height / height
                    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                    
                    if scale < 1.0:
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        annotated_frame = cv2.resize(annotated_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Encode frame for streaming
                    ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error in intelligent analysis: {e}")
                    # Fallback to basic frame with proper aspect ratio
                    cv2.putText(frame, "ANALYSIS ERROR", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Apply same aspect ratio handling
                    height, width = frame.shape[:2]
                    max_width, max_height = 1280, 720
                    scale_w = max_width / width
                    scale_h = max_height / height
                    scale = min(scale_w, scale_h, 1.0)
                    
                    if scale < 1.0:
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with intelligent analysis"""
    return Response(generate_intelligent_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    """Get current alerts API"""
    recent_alerts = shared_state.get_recent_alerts(20)
    return jsonify({
        'alerts': recent_alerts,
        'total': len(recent_alerts),
        'unacknowledged': len([a for a in recent_alerts if not a['acknowledged']])
    })

@app.route('/api/stats')
def get_stats():
    """Get system statistics API"""
    # Update FPS calculation
    current_time = datetime.now()
    time_diff = (current_time - shared_state.last_update).total_seconds()
    if time_diff > 0:
        shared_state.stats['current_fps'] = min(30, shared_state.stats['total_frames'] / time_diff)
    
    return jsonify(shared_state.stats)

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    if shared_state.acknowledge_alert(alert_id):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Alert not found'}), 404

@app.route('/api/test_alert', methods=['POST'])
def test_alert():
    """Generate a test alert (for demonstration)"""
    test_alerts = [
        {
            'rule': 'unattended_bag',
            'severity': 'high',
            'message': 'Unattended bag detected in waiting area',
            'object_id': 'bag_127',
            'location': 'Main Hall - Zone A'
        },
        {
            'rule': 'restricted_zone',
            'severity': 'medium', 
            'message': 'Unauthorized access in restricted area',
            'object_id': 'person_42',
            'location': 'Server Room - Restricted Zone'
        },
        {
            'rule': 'loitering',
            'severity': 'medium',
            'message': 'Suspicious loitering detected',
            'object_id': 'person_15',
            'location': 'Entrance Hall - Security Zone'
        }
    ]
    
    import random
    alert_data = random.choice(test_alerts)
    shared_state.add_alert(alert_data)
    
    return jsonify({'success': True, 'alert': alert_data})  

if __name__ == '__main__':
    print("🚀 Starting EP-SAD Security Dashboard with Intelligent System...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🎯 Features:")
    print("   - Live video feed with REAL intelligent analysis")
    print("   - YOLO object detection and tracking")
    print("   - Sequence logic engine for complex event detection")
    print("   - Real-time alert monitoring from actual model")
    print("   - Alert acknowledgment system")
    print("   - System statistics and analytics")
    print("🧠 Intelligent System: ACTIVE")
    print("   - Unattended bag detection")
    print("   - Restricted zone violations")
    print("   - Suspicious loitering detection")
    print("   - Object tracking and behavior analysis")
    print("-" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)