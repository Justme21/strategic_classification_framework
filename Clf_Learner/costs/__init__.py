from ..interfaces import BaseCost

from .quadratic_cost import QuadraticCost

COST_DICT: dict[str, type[BaseCost]] = {
    "quadratic": QuadraticCost
}