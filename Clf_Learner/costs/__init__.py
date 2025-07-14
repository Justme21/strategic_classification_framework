from ..interfaces import BaseCost

from .quadratic_cost import QuadraticCost
from .zero_cost import ZeroCost

COST_DICT: dict[str, type[BaseCost]] = {
    "quadratic": QuadraticCost,
    "zero": ZeroCost
}