from django.dispatch import Signal

# Provides arguments 'task_id', 'data', 'metadata'
measurement_action_completed = Signal()

# Provides arguments: 'latitude', 'longitude'
location_action_completed = Signal()

trigger_api_restart = Signal()

# Provides argument: 'component'
register_component_with_status = Signal()

# Provides argument: signal_analyzer
register_signal_analyzer = Signal()

# Provides argument: sensor
register_sensor = Signal()
