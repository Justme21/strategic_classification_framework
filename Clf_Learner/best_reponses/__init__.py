from ..interfaces import BaseBestResponse

from .augmented_lagrangian_best_response import AugmentedLagrangianBestResponse
from .identity_best_response import IdentityResponse
from .lagrangian_best_response import LagrangianBestResponse
from .linear_best_response import LinearBestResponse
from .sgd_best_response import SGDBestResponse

BR_DICT: dict[str, type[BaseBestResponse]] = {
  "augmented_lagrange": AugmentedLagrangianBestResponse,
  "identity": IdentityResponse,
  "lagrange": LagrangianBestResponse,
  "linear": LinearBestResponse,
  "sgd": SGDBestResponse,
}