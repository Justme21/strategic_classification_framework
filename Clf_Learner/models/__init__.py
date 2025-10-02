from ..interfaces import BaseModel
from .iterated_randomised_model import IteratedRandomisedModel
from .linear_model import LinearModel
from .mlp_model import MLPModel
from .randomised_model import RandomisedModel
from .parabolic_model import ParabolicModel
from .quadratic_model import QuadraticModel

MODEL_DICT: dict[str, type[BaseModel]] = {
    "iterated_randomised": IteratedRandomisedModel,
    "linear": LinearModel,
    "mlp": MLPModel,
    "parabolic": ParabolicModel,
    "quadratic": QuadraticModel,
    "randomised": RandomisedModel,
}


