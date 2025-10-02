from ..losses.loss_tools import ImplicitDifferentiationLossWrapper
from .specs import SPEC_DICT
from .results_tools import get_results_directory

from ..best_reponses import BR_DICT
from ..costs import COST_DICT
from ..losses import LOSS_DICT
from ..models import MODEL_DICT
from ..utilities import UTILITY_DICT

from ..interfaces import BaseDataset

from ..tools.device_tools import get_device

DEVICE = get_device()


def _build_model_from_spec(model_spec:dict, init_args:dict, comp_args:dict, result_addr:str, dataset:BaseDataset, implicit:bool):
    # A model might not require a cost or a utility. 
    # By assumption ever model should require a best response
    cost = COST_DICT.get(model_spec['cost'])
    utility = UTILITY_DICT.get(model_spec['utility'])
    best_response = BR_DICT[model_spec['best_response']]
    loss = LOSS_DICT[model_spec['loss']]
    model = MODEL_DICT[model_spec['model']]

    if cost is not None:
        cost = cost(**init_args, **comp_args.get('cost', {}))
        cost.set_standardiser(dataset.get_standardiser())
    if utility is not None:
        utility = utility(**init_args, **comp_args.get('utility', {}))

    strategic_columns = dataset.get_strategic_columns()
    best_response = best_response(cost=cost, utility=utility, strategic_columns=strategic_columns, **init_args, **comp_args.get('best_response',{}))
    loss = loss(**init_args, **comp_args.get('loss', {}))

    if implicit:
        loss = ImplicitDifferentiationLossWrapper(loss)

    model_addr = get_results_directory(result_addr, dataset.filename, model_spec )
    model = model(best_response=best_response, loss=loss, address=model_addr, **init_args, **comp_args.get('model', {}))

    return model

def get_model(model_spec, result_addr, dataset, implicit, init_args={}, comp_args={}):
    model = _build_model_from_spec(model_spec=model_spec, init_args=init_args, comp_args=comp_args, result_addr=result_addr, dataset=dataset, implicit=implicit)
    model.to(DEVICE)

    return model

def get_model_spec(model_spec_name=None, br_name=None, cost_name=None, loss_name=None, model_type_name=None, utility_name=None):
    assert model_spec_name is not None or any([x is not None for x in [br_name, cost_name, model_type_name, utility_name]])

    if model_spec_name is not None:
        assert model_spec_name in SPEC_DICT
        model_spec = SPEC_DICT[model_spec_name]
    else:
        model_spec = {"model": model_type_name, "best_response": br_name, "cost": cost_name, "loss": loss_name, "utility": utility_name}

    return model_spec