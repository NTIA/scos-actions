from django.dispatch import Signal

# Provides arguments 'task_id', 'data', 'metadata'
measurement_action_completed = Signal()

# Provides arguments: 'latitude', 'longitude'
location_action_completed = Signal()

trigger_api_restart = Signal()
