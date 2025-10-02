from ..interfaces import BaseLoss

#from .cost_augmented_ssvm_hinge_loss import CostAugmentedStrategicHingeLoss
from .strategic_cross_entropy_loss import StrategicCrossEntropyLoss
from .ssvm_hinge_loss import StrategicSVMHingeLoss
from .naive_ssvm_hinge_loss import NaiveStrategicSVMHingeLoss
#from .linear_plus_logistic_loss import LinearPlusLogisticLoss
#from .linear_plus_kl_loss import LinearPlusKLLoss

LOSS_DICT: dict[str, type[BaseLoss]] = {
    #"cost_ssvm_hinge": CostAugmentedStrategicHingeLoss,
    "strategic_cross_entropy": StrategicCrossEntropyLoss,
    "ssvm_hinge": StrategicSVMHingeLoss,
    "naive_ssvm_hinge": NaiveStrategicSVMHingeLoss,
    #"lin_plus_log": LinearPlusLogisticLoss,
    #"lin_plus_kl": LinearPlusKLLoss
}