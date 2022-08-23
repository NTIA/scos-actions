from django.dispatch import Signal


measurement_action_completed = Signal(
    providing_args=["task_id", "data", "metadata"]
)
location_action_completed = Signal(
    providing_args=["latitude", "longitude"]
)
monitor_action_completed = Signal(providing_args=["sigan_healthy"])

register_component_with_status = Signal(providing_args=["component"])
