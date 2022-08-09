from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import switches


class SetSwitchStates(Action):

    def __init__(self, parameters, sigan=mock_sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)

    def __call__(self, schedule_entry, task_id):
        switch_id = self.parameter_map['switch_id']
        states = self.parameter_map['states']
        switch = switches[switch_id]
        for state in states:
            switch.set_state(state)
