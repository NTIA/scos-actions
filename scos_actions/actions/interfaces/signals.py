from django.dispatch import Signal

# Provides arguments 'task_id', 'data', 'metadata'
measurement_action_completed = Signal()

# Provides arguments: 'latitude', 'longitude'
location_action_completed = Signal()

# Provides arguments: 'sigan_healthy'
monitor_action_completed = Signal()

# Provides argument: 'component'
register_component_with_status = Signal()
