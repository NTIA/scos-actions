import logging
from os import path
from pathlib import Path

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


#def load_from_yaml(action_classes, yaml_dir=ACTION_DEFINITIONS_DIR):
def load_from_yaml(action_classes, yaml_dir=path.join(path.dirname(path.dirname(path.abspath(__file__))), "configs/actions")):
    """Load any YAML files in yaml_dir."""
    parsed_actions = {}
    yaml = YAML(typ="safe")
    yaml_path = Path(yaml_dir)
    for yaml_file in yaml_path.glob("*.yml"):
        definition = yaml.load(yaml_file)
        for class_name, parameters in definition.items():
            try:
                action = action_classes[class_name](**parameters)
                parsed_actions[action.name] = action
            except KeyError as exc:
                err = "Nonexistent action class name {!r} referenced in {!r}"
                logger.error(err.format(class_name, yaml_file.name))
                logger.exception(exc)
                raise exc
            except TypeError as exc:
                err = "Invalid parameter list {!r} referenced in {!r}"
                logger.error(err.format(parameters, yaml_file.name))
                logger.exception(exc)
                raise exc
    return parsed_actions
