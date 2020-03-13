import django.dispatch

measurement_action_completed = django.dispatch.Signal(providing_args=["task_id", "data", "metadata"])
location_action_completed = django.dispatch.Signal(providing_args=["latitude", "longitude"])