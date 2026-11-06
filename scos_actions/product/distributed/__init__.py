"""Ray-based execution of the SEA data products (tasks, actors, supervisors).

This subpackage depends on the product classes in ``scos_actions.product``;
those classes must never import from here.
"""

from .actors import DataProductActor
from .iq_processor import (
    IQProcessorProductActors,
)
