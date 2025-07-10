from ..interfaces import BaseUtility

from .linear_utility import LinearUtility
from .strategic_utility import StrategicUtility

UTILITY_DICT: dict[str, type[BaseUtility]]= {
    "linear": LinearUtility,
    "strategic": StrategicUtility,
}