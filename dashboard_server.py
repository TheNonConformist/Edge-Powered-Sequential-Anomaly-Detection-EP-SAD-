from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import json
import threading
import time
from datetime import datetime
from ultralytics import YOLO

# Import our intelligent system (simplified version for demo)
class DashboardSystem:
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
    
    def add_alert(self, alert_data):
        """Add a new alert to the dashboard"""
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': alert_data.get('type', 'Unknown'),
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
        self.video_frame = frame
        self.stats['total_frames'] += 1
    
    def get_recent_alerts(self, count=10):
        """Get most recent alerts"""
        return self.alerts[:count]
    
    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False

# Initialize Flask app and dashboard system
app = Flask(__name__)
dashboard_system = DashboardSystem()

# Sample camera feed generator (replace with your actual video source)
def generate_sample_feed():
    """Generate a sample video feed for demonstration"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Create a synthetic video feed if no camera available
        print("No camera found, creating synthetic feed...")
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
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    else:
        print("Camera found, using live feed...")
        while True:
            ret, frame = cap.read()
            if ret:
                # Add timestamp to frame
                cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)

def generate_annotated_feed():
    """Generate video feed with intelligent annotations"""
    # This would integrate with your actual intelligent system
    # For now, we'll use a sample feed
    return generate_sample_feed()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_annotated_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    """Get current alerts API"""
    recent_alerts = dashboard_system.get_recent_alerts(20)
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
    time_diff = (current_time - dashboard_system.last_update).total_seconds()
    if time_diff > 0:
        dashboard_system.stats['current_fps'] = min(30, dashboard_system.stats['total_frames'] / time_diff)
    
    return jsonify(dashboard_system.stats)

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    if dashboard_system.acknowledge_alert(alert_id):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Alert not found'}), 404

@app.route('/api/test_alert', methods=['POST'])
def test_alert():
    """Generate a test alert (for demonstration)"""
    test_alerts = [
        {
            'type': 'unattended_bag',
            'severity': 'high',
            'message': 'Unattended bag detected in waiting area',
            'object_id': 'bag_127',
            'location': 'Main Hall - Zone A'
        },
        {
            'type': 'restricted_zone',
            'severity': 'medium', 
            'message': 'Unauthorized access in restricted area',
            'object_id': 'person_42',
            'location': 'Server Room - Restricted Zone'
        },
        {
            'type': 'loitering',
            'severity': 'medium',
            'message': 'Suspicious loitering detected',
            'object_id': 'person_15',
            'location': 'Entrance Hall - Security Zone'
        }
    ]
    
    import random
    alert_data = random.choice(test_alerts)
    dashboard_system.add_alert(alert_data)
    
    return jsonify({'success': True, 'alert': alert_data})

# Background thread to simulate intelligent system updates
def simulate_intelligent_system():
    """Simulate the intelligent system generating alerts"""
    alert_types = [
        ('unattended_bag', 'high', 'Unattended bag detected', 'Main Hall'),
        ('restricted_zone', 'medium', 'Zone violation', 'Restricted Area'),
        ('loitering', 'medium', 'Suspicious loitering', 'Entrance'),
        ('crowding', 'low', 'High density area', 'Waiting Zone')
    ]
    
    while True:
        # Randomly generate alerts (less frequently)
        if np.random.random() < 0.02:  # 2% chance per iteration
            alert_type, severity, base_message, location = np.random.choice(
                len(alert_types), p=[0.4, 0.3, 0.2, 0.1]
            )
            alert_type, severity, base_message, location = alert_types[alert_type]
            
            alert_data = {
                'type': alert_type,
                'severity': severity,
                'message': f"{base_message} in {location}",
                'object_id': f"{alert_type}_{np.random.randint(100, 999)}",
                'location': location
            }
            dashboard_system.add_alert(alert_data)
            print(f"Simulated alert: {alert_data['message']}")
        
        # Update stats
        dashboard_system.stats['active_tracks'] = np.random.randint(0, 8)
        time.sleep(1)  

if __name__ == '__main__':
    # Start background simulation thread
    simulation_thread = threading.Thread(target=simulate_intelligent_system, daemon=True)
    simulation_thread.start()
    
    print("🚀 Starting EP-SAD Security Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🎯 Features:")
    print("   - Live video feed with intelligent annotations")
    print("   - Real-time alert monitoring")
    print("   - Alert acknowledgment system")
    print("   - System statistics and analytics")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)