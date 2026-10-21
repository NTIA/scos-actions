"""Ray-based execution of the SEA data products (tasks, actors, supervisors).

This subpackage depends on the product classes in ``scos_actions.product``;
those classes must never import from here.
"""

from .actors import ProductActor
from .iq_processor import (
    IQProcessorActors,
    IQProcessorLocal,
    IQProcessorProductActors,
    IQProcessorTasks,
    IQProcessorTasks2,
)
from .tasks import (
    PRODUCT_NAMES,
    build_local_products,
    build_product_ref,
    build_products,
    compute_product,
    product_key,
)
