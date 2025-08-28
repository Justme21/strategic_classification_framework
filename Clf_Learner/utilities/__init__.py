from ..interfaces import BaseUtility

from .linear_utility import LinearUtility
from .strategic_utility import StrategicUtility
from .strategic_sigmoid_utility import StrategicSigmoidUtility

UTILITY_DICT: dict[str, type[BaseUtility]]= {
    "linear": LinearUtility,
    "strategic": StrategicUtility,
    "strategic_sigmoid": StrategicSigmoidUtility,
}