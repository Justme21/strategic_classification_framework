from ..interfaces import BaseLoss

from .ssvm_hinge_loss import StrategicSVMHingeLoss
from .naive_ssvm_hinge_loss import NaiveStrategicSVMHingeLoss
from .linear_plus_logistic_loss import LinearPlusLogisticLoss
from .linear_plus_kl_loss import LinearPlusKLLoss

LOSS_DICT: dict[str, type[BaseLoss]] = {
    "ssvm_hinge": StrategicSVMHingeLoss,
    "naive_ssvm_hinge": NaiveStrategicSVMHingeLoss,
    #"lin_plus_log": LinearPlusLogisticLoss,
    #"lin_plus_kl": LinearPlusKLLoss
}