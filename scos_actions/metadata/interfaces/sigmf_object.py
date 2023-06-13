from dataclasses import dataclass
from typing import Any


@dataclass
class SigMFObject:
    def __post_init__(self):
        self.object_type = self.__class__.__name__
        self.json_obj, self.obj_keys = {}, {}
        self.required_err_msg = (
            f"{self.object_type} objects require a value to be specified for "
        )

    def check_required(self, value: Any, keyname: str) -> None:
        assert value is not None, self.required_err_msg + keyname

    def create_json_object(self) -> None:
        meta_vars = vars(self)
        for varname, value in meta_vars.items():
            if value is not None:
                try:
                    sigmf_key = meta_vars["obj_keys"][varname]
                    try:
                        # Handles case when value is a custom SigMF object
                        self.json_obj[sigmf_key] = value.json_obj
                    except AttributeError:
                        # Handles standard data types
                        self.json_obj[sigmf_key] = value
                except KeyError:
                    pass
        return
