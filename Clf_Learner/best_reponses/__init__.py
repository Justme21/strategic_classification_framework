from ..interfaces import BaseBestResponse

from .augmented_lagrangian_best_response import AugmentedLagrangianBestResponse
from .identity_best_response import IdentityResponse
from .lagrangian_best_response import LagrangianBestResponse
#from .lagrangian_best_response_alt import AltLagrangianBestResponse
from .linear_best_response import LinearBestResponse
#from .optimiser_best_response import OptimiserBestResponse
from .gradient_ascent_best_response import GradientAscentBestResponse

BR_DICT: dict[str, type[BaseBestResponse]] = {
    #"alt_lagrange": AltLagrangianBestResponse,
    "augmented_lagrange": AugmentedLagrangianBestResponse,
    "gradient": GradientAscentBestResponse,
    "identity": IdentityResponse,
    "lagrange": LagrangianBestResponse,
    "linear": LinearBestResponse,
    #"optimiser": OptimiserBestResponse,
}