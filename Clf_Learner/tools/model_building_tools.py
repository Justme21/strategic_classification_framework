from .loss_tools import ImplicitDifferentiationLossWrapper
from .specs import SPEC_DICT
from .results_tools import get_results_directory

from ..best_reponses import BR_DICT
from ..costs import COST_DICT
from ..losses import LOSS_DICT
from ..models import MODEL_DICT
from ..utilities import UTILITY_DICT


def _build_model_from_spec(model_spec, init_args, comp_args, result_addr, dataset_filename):
    # A model might not require a cost or a utility. 
    # By assumption ever model should require a best response
    cost = COST_DICT.get(model_spec['cost'])
    utility = UTILITY_DICT.get(model_spec['utility'])
    best_response = BR_DICT[model_spec['best_response']]
    loss = LOSS_DICT[model_spec['loss']]
    model = MODEL_DICT[model_spec['model']]

    if cost is not None:
        cost = cost(**init_args, **comp_args.get('cost', {}))
    if utility is not None:
        utility = utility(**init_args, **comp_args.get('utility', {}))
    best_response = best_response(cost=cost, utility=utility, **init_args, **comp_args.get('best_response',{}))
    loss = loss(**init_args, **comp_args.get('loss', {}))

    #loss = ImplicitDifferentiationLossWrapper(loss)

    model_addr = get_results_directory(result_addr, dataset_filename, model_spec )
    model = model(best_response=best_response, loss=loss, address=model_addr, **init_args, **comp_args.get('model', {}))

    return model

def get_model(model_spec, result_addr, dataset_filename, init_args={}, comp_args={}):
    model = _build_model_from_spec(model_spec, init_args, comp_args, result_addr, dataset_filename)

    return model

def get_model_spec(model_spec_name=None, br_name=None, cost_name=None, loss_name=None, model_type_name=None, utility_name=None):
    assert model_spec_name is not None or any([x is not None for x in [br_name, cost_name, model_type_name, utility_name]])

    if model_spec_name is not None:
        assert model_spec_name in SPEC_DICT
        model_spec = SPEC_DICT[model_spec_name]
    else:
        model_spec = {"model": model_type_name, "best_response": br_name, "cost": cost_name, "loss": loss_name, "utility": utility_name}

    return model_spec