from ..interfaces import BaseUtility

from .binary_strategic_utility import BinaryStrategicUtility
from .expected_hinge_utility import ExpectedHingeUtility
from .hinge_utility import HingeUtility
from .randomised_strategic_utility import RandomisedStrategicUtility
from .strategic_utility import StrategicUtility
from .strategic_sigmoid_utility import StrategicSigmoidUtility
from .strategic_tanh_utility import StrategicTanhUtility

UTILITY_DICT: dict[str, type[BaseUtility]]= {
    "binary_strategic": BinaryStrategicUtility,
    "expected_hinge": ExpectedHingeUtility,
    "hinge": HingeUtility,
    "randomised_strategic": RandomisedStrategicUtility,
    "strategic": StrategicUtility,
    "strategic_sigmoid": StrategicSigmoidUtility,
    "strategic_tanh": StrategicTanhUtility,
}