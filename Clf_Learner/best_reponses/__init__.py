from ..interfaces import BaseBestResponse

from .identity_best_response import IdentityResponse
from .linear_best_response import LinearBestResponse
from .sgd_best_response import SGDBestResponse

BR_DICT: dict[str, type[BaseBestResponse]] = {
  "identity": IdentityResponse,
  "linear": LinearBestResponse,
  "sgd": SGDBestResponse,
}