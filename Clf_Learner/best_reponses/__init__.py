from ..interfaces import BaseBestResponse
from .identity_best_response import IdentityResponse
from .linear_best_response import LinearBestResponse

BR_DICT: dict[str, type[BaseBestResponse]] = {
  "identity": IdentityResponse,
  "linear": LinearBestResponse,
}