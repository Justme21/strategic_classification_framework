from ..interfaces import BaseModel
from .iterated_randomised_model import IteratedRandomisedModel
from .linear_model import LinearModel
from .randomised_model import RandomisedModel
from .parabolic_model import ParabolicModel

MODEL_DICT: dict[str, type[BaseModel]] = {
    "iterated_randomised": IteratedRandomisedModel,
    "linear": LinearModel,
    "randomised": RandomisedModel,
    "parabolic": ParabolicModel,
}


