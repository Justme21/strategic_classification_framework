from ..interfaces import BaseModel
from .linear_model import LinearModel
from .randomised_linear_model import RandomisedLinearModel

MODEL_DICT: dict[str, type[BaseModel]] = {
    "linear": LinearModel,
    "randomised_linear": RandomisedLinearModel
}


