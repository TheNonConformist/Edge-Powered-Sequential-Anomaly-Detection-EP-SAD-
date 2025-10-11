import time
import json
from datetime import datetime

class SequenceLogicEngine:
    """
    Smart rule-based state machine for detecting complex event sequences
    """
    def __init__(self):
        self.rules = {}
        self.object_states = {}  # Track state for each object
        self.sequence_history = {}  # Track sequence progress
        self.triggered_alerts = []  # Store all triggered alerts
        
        # Define the rules for different scenarios
        self._initialize_rules()
        
        print("🧠 Sequence Logic Engine initialized!")
        print("✓ Monitoring: Unattended bags, Zone violations, Suspicious behavior")
    
    def _initialize_rules(self):
        """Define the detection rules and sequences"""
        
        # Rule 1: Unattended Bag Sequence
        self.rules['unattended_bag'] = {
            'name': 'Unattended Bag Detection',
            'description': 'Person places bag and leaves without it',
            'sequence': [
                {'state': 'PERSON_NEAR_BAG', 'description': 'Person carrying or near bag'},
                {'state': 'BAG_STATIONARY', 'description': 'Bag becomes stationary'}, 
                {'state': 'PERSON_LEFT', 'description': 'Person moves away from bag'},
                {'state': 'TIME_ELAPSED', 'description': 'Bag remains unattended for 30 seconds'}
            ],
            'alert_message': '🚨 UNATTENDED BAG: Bag appears to be left unattended',
            'severity': 'HIGH'
        }
        
        # Rule 2: Restricted Zone Violation
        self.rules['restricted_zone'] = {
            'name': 'Restricted Zone Violation',
            'description': 'Object enters restricted zone',
            'sequence': [
                {'state': 'IN_RESTRICTED_ZONE', 'description': 'Object enters restricted zone'},
                {'state': 'TIME_IN_ZONE', 'description': 'Object remains in zone for 10 seconds'}
            ],
            'alert_message': '🚨 RESTRICTED ZONE: Object in restricted area',
            'severity': 'MEDIUM'
        }
        
        # Rule 3: Suspicious Loitering
        self.rules['loitering'] = {
            'name': 'Suspicious Loitering',
            'description': 'Person lingers in sensitive area',
            'sequence': [
                {'state': 'IN_SENSITIVE_ZONE', 'description': 'Person in sensitive zone'},
                {'state': 'STATIONARY_IN_ZONE', 'description': 'Person remains stationary'},
                {'state': 'EXTENDED_PRESENCE', 'description': 'Person stays for 60 seconds'}
            ],
            'alert_message': '🚨 LOITERING: Person lingering in sensitive area',
            'severity': 'MEDIUM'
        }
        
        # Rule 4: Object Transfer
        self.rules['object_transfer'] = {
            'name': 'Suspicious Object Transfer', 
            'description': 'Object transferred between people in sensitive area',
            'sequence': [
                {'state': 'PERSONS_MEETING', 'description': 'Two people meet'},
                {'state': 'OBJECT_TRANSFER', 'description': 'Object changes hands'},
                {'state': 'SEPARATION', 'description': 'People separate quickly'}
            ],
            'alert_message': '🚨 SUSPICIOUS TRANSFER: Object exchanged between people',
            'severity': 'HIGH'
        }
    
    def update_object_state(self, track_id, object_type, properties):
        """
        Update state for a specific object and check rules
        properties: dict with current state info like {
            'position': (x, y),
            'velocity': speed,
            'zone': zone_name,
            'nearby_objects': [list of nearby track_ids],
            'stationary': True/False,
            'interactions': [list of interactions]
        }
        """
        current_time = time.time()
        
        # Initialize object state if not exists
        if track_id not in self.object_states:
            self.object_states[track_id] = {
                'type': object_type,
                'states': {},
                'history': [],
                'created_time': current_time,
                'last_update': current_time
            }
        
        # Update object properties
        obj_state = self.object_states[track_id]
        obj_state['last_update'] = current_time
        obj_state['history'].append({
            'time': current_time,
            'properties': properties
        })
        
        # Keep only recent history (last 5 minutes)
        obj_state['history'] = [h for h in obj_state['history'] 
                              if current_time - h['time'] < 300]
        
        # Update specific states based on properties
        self._update_states_from_properties(track_id, properties)
        
        # Check all rules for this object
        alerts = self._check_rules(track_id)
        
        return alerts
    
    def _update_states_from_properties(self, track_id, properties):
        """Update object states based on current properties"""
        obj_state = self.object_states[track_id]
        
        # State: In restricted zone
        if properties.get('zone') == 'RESTRICTED AREA':
            obj_state['states']['IN_RESTRICTED_ZONE'] = True
            # Track time in zone
            if 'zone_entry_time' not in obj_state:
                obj_state['zone_entry_time'] = time.time()
        else:
            obj_state['states']['IN_RESTRICTED_ZONE'] = False
            obj_state.pop('zone_entry_time', None)
        
        # State: Stationary
        obj_state['states']['STATIONARY'] = properties.get('stationary', False)
        
        # State: Near other objects
        nearby_objects = properties.get('nearby_objects', [])
        obj_state['states']['NEAR_OTHER_OBJECTS'] = len(nearby_objects) > 0
        
        # Check if near a bag (for unattended bag detection)
        if obj_state['type'] == 'person':
            nearby_bags = [obj_id for obj_id in nearby_objects 
                          if self.object_states.get(obj_id, {}).get('type') in 
                          ['backpack', 'handbag', 'suitcase', 'bag']]
            obj_state['states']['NEAR_BAG'] = len(nearby_bags) > 0
            if nearby_bags:
                obj_state['nearby_bag'] = nearby_bags[0]  # Track the specific bag
    
    def _check_rules(self, track_id):
        """Check all rules against current object state"""
        alerts = []
        current_time = time.time()
        obj_state = self.object_states[track_id]
        
        # Rule 1: Unattended Bag Detection
        if obj_state['type'] in ['backpack', 'handbag', 'suitcase', 'bag']:
            alert = self._check_unattended_bag(track_id, current_time)
            if alert:
                alerts.append(alert)
        
        # Rule 2: Restricted Zone Violation  
        if obj_state['type'] == 'person':
            alert = self._check_restricted_zone(track_id, current_time)
            if alert:
                alerts.append(alert)
            
            # Rule 3: Loitering
            alert = self._check_loitering(track_id, current_time)
            if alert:
                alerts.append(alert)
        
        # Rule 4: Object Transfer (check interactions)
        alert = self._check_object_transfer(track_id, current_time)
        if alert:
            alerts.append(alert)
        
        return alerts
    
    def _check_unattended_bag(self, bag_id, current_time):
        """Check for unattended bag sequence"""
        bag_state = self.object_states[bag_id]
        
        # Initialize sequence tracking
        if bag_id not in self.sequence_history:
            self.sequence_history[bag_id] = {
                'rule': 'unattended_bag',
                'current_step': 0,
                'step_start_time': current_time,
                'metadata': {}
            }
        
        seq = self.sequence_history[bag_id]
        rule = self.rules['unattended_bag']
        
        # Step 1: Person near bag
        if seq['current_step'] == 0:
            if bag_state['states'].get('NEAR_OTHER_OBJECTS'):
                nearby_people = [obj_id for obj_id in bag_state.get('nearby_objects', [])
                               if self.object_states.get(obj_id, {}).get('type') == 'person']
                if nearby_people:
                    seq['current_step'] = 1
                    seq['step_start_time'] = current_time
                    seq['metadata']['person_id'] = nearby_people[0]
                    print(f"🔍 Sequence: Bag {bag_id} being carried by person {nearby_people[0]}")
        
        # Step 2: Bag becomes stationary
        elif seq['current_step'] == 1:
            if bag_state['states'].get('STATIONARY'):
                if current_time - seq['step_start_time'] > 5:  # Stationary for 5 seconds
                    seq['current_step'] = 2
                    seq['step_start_time'] = current_time
                    print(f"🔍 Sequence: Bag {bag_id} placed down and stationary")
        
        # Step 3: Person leaves bag
        elif seq['current_step'] == 2:
            person_id = seq['metadata'].get('person_id')
            if person_id and person_id in self.object_states:
                person_state = self.object_states[person_id]
                # Check if person is no longer near the bag
                if bag_id not in person_state.get('nearby_objects', []):
                    seq['current_step'] = 3
                    seq['step_start_time'] = current_time
                    print(f"🔍 Sequence: Person {person_id} left bag {bag_id}")
        
        # Step 4: Time elapsed (bag remains unattended)
        elif seq['current_step'] == 3:
            if current_time - seq['step_start_time'] > 30:  # 30 seconds unattended
                # TRIGGER ALERT!
                alert = self._create_alert('unattended_bag', bag_id, seq['metadata'])
                seq['current_step'] = 0  # Reset sequence
                return alert
        
        return None
    
    def _check_restricted_zone(self, person_id, current_time):
        """Check for restricted zone violation"""
        person_state = self.object_states[person_id]
        
        if person_state['states'].get('IN_RESTRICTED_ZONE'):
            zone_entry_time = person_state.get('zone_entry_time', current_time)
            
            if current_time - zone_entry_time > 10:  # 10 seconds in restricted zone
                # TRIGGER ALERT!
                alert = self._create_alert('restricted_zone', person_id, {
                    'zone_entry_time': zone_entry_time,
                    'duration': current_time - zone_entry_time
                })
                # Reset zone entry time to prevent repeated alerts
                person_state['zone_entry_time'] = current_time
                return alert
        
        return None
    
    def _check_loitering(self, person_id, current_time):
        """Check for suspicious loitering"""
        person_state = self.object_states[person_id]
        
        if (person_state['states'].get('IN_RESTRICTED_ZONE') and 
            person_state['states'].get('STATIONARY')):
            
            zone_entry_time = person_state.get('zone_entry_time', current_time)
            
            if current_time - zone_entry_time > 60:  # 60 seconds stationary in zone
                alert = self._create_alert('loitering', person_id, {
                    'zone_entry_time': zone_entry_time,
                    'duration': current_time - zone_entry_time
                })
                person_state['zone_entry_time'] = current_time  # Reset
                return alert
        
        return None
    
    def _check_object_transfer(self, track_id, current_time):
        """Check for suspicious object transfers"""
        # This would require tracking object ownership changes
        # For now, we'll implement a simplified version
        return None
    
    def _create_alert(self, rule_name, object_id, metadata):
        """Create an alert object"""
        rule = self.rules[rule_name]
        
        alert = {
            'id': f"alert_{len(self.triggered_alerts) + 1}",
            'rule': rule_name,
            'rule_name': rule['name'],
            'object_id': object_id,
            'object_type': self.object_states[object_id]['type'],
            'message': rule['alert_message'],
            'severity': rule['severity'],
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        self.triggered_alerts.append(alert)
        
        # Print to console
        print(f"🚨🚨🚨 {alert['message']}")
        print(f"       Object: {alert['object_type']} {alert['object_id']}")
        print(f"       Severity: {alert['severity']}")
        print(f"       Time: {alert['timestamp']}")
        
        return alert
    
    def get_active_alerts(self):
        """Get all active alerts (last 10 minutes)"""
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.triggered_alerts
            if current_time - datetime.fromisoformat(alert['timestamp']).timestamp() < 600
        ]
        return recent_alerts
    
    def get_rule_statistics(self):
        """Get statistics about rule triggers"""
        stats = {}
        for rule_name in self.rules:
            rule_alerts = [a for a in self.triggered_alerts if a['rule'] == rule_name]
            stats[rule_name] = {
                'total_alerts': len(rule_alerts),
                'recent_alerts': len([a for a in rule_alerts 
                                    if time.time() - datetime.fromisoformat(a['timestamp']).timestamp() < 300])
            }
        return stats