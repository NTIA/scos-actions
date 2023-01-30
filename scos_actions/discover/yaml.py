import logging
from pathlib import Path

from ruamel.yaml import YAML

from scos_actions.settings import ACTION_DEFINITIONS_DIR

logger = logging.getLogger(__name__)


def load_from_yaml(action_classes, sigan, gps, yaml_dir: Path = ACTION_DEFINITIONS_DIR):
    """Load any YAML files in yaml_dir."""
    parsed_actions = {}
    yaml = YAML(typ="safe")
    yaml_path = Path(yaml_dir)
    for yaml_file in yaml_path.glob("*.yml"):
        definition = yaml.load(yaml_file)
        for class_name, parameters in definition.items():
            try:
                logger.debug("Attempting to configure: " + class_name)
                action = action_classes[class_name](
                    parameters=parameters, sigan=sigan, gps=gps
                )
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
            except Exception as exc:
                logger.error("Unable to load yaml:", exc)
                raise exc
    return parsed_actions
