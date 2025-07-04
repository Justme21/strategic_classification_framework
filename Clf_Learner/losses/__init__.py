from ..interfaces import BaseLoss
from .ssvm_hinge_loss import StrategicSVMHingeLoss
from .naive_ssvm_hing_loss import NaiveStrategicSVMHingeLoss

LOSS_DICT: dict[str, type[BaseLoss]] = {
    "ssvm_hinge": StrategicSVMHingeLoss,
    "naive_ssvm_hinge": NaiveStrategicSVMHingeLoss
}