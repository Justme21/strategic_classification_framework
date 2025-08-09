from ..interfaces import BaseModel
from .linear_model import LinearModel
from .parabolic_model import ParabolicModel

MODEL_DICT = {
    "linear": LinearModel,
    "parabolic": ParabolicModel,
}


