from ..interfaces import BaseModel
from .linear_model import LinearModel
from .randomised_model import RandomisedModel
from .randomised_linear_delta_model import RandomisedLinearDeltaModel
from .randomised_linear_model import RandomisedLinearModel
from .parabolic_model import ParabolicModel

MODEL_DICT: dict[str, type[BaseModel]] = {
    "linear": LinearModel,
    "randomised": RandomisedModel,
    "randomised_linear": RandomisedLinearModel,
    "randomised_linear_delta": RandomisedLinearDeltaModel,
    "parabolic": ParabolicModel,
}


