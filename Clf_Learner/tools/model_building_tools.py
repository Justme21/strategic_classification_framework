from .specs import SPEC_DICT

from ..best_reponses import BR_DICT
from ..costs import COST_DICT
from ..losses import LOSS_DICT
from ..models import MODEL_DICT
from ..utilities import UTILITY_DICT
from .implicit_grad_tool import hyper_grad_fuc


def _build_model_from_spec(model_spec, init_args, comp_args):
	# A model might not require a cost or a utility.
	# By assumption ever model should require a best response
	cost = COST_DICT.get(model_spec['cost'])
	utility = UTILITY_DICT.get(model_spec['utility'])
	best_response = BR_DICT[model_spec['best_response']]
	loss = LOSS_DICT[model_spec['loss']]
	model = MODEL_DICT[model_spec['model']]
	hyper_lr = 0.1
	hyper_truncate_step = 20
	
	if cost is not None:
		cost = cost(**init_args, **comp_args.get('cost', {}))
	if utility is not None:
		utility = utility(**init_args, **comp_args.get('utility', {}))
	
	best_response = best_response(cost=cost, utility=utility, **init_args, **comp_args.get('best_response', {}))
	
	loss = loss(**init_args, **comp_args.get('loss', {}))
	
	model = model(best_response=best_response, loss=loss, **init_args, **comp_args.get('model', {}))
	
	# x and y are the input-label pair
	# I assume best_reponse
	inner_loss = model.object(model(x), y)
	outer_loss = loss(model(x), y)
	
	hyper_grad = hyper_grad_fuc(inner_loss, outer_loss, model.get_weights(), best_response.get_weights(),
	                            learning_rate=hyper_lr, truncate_iter=hyper_truncate_step)
	
	return model


def get_model(model_spec, init_args={}, comp_args={}):
	model = _build_model_from_spec(model_spec, init_args, comp_args)
	
	return model


def get_model_spec(model_spec_name=None, br_name=None, cost_name=None, loss_name=None, model_type_name=None,
                   utility_name=None):
	assert model_spec_name is not None or any(
		[x is not None for x in [br_name, cost_name, model_type_name, utility_name]])
	
	if model_spec_name is not None:
		assert model_spec_name in SPEC_DICT
		model_spec = SPEC_DICT[model_spec_name]
	else:
		model_spec = {"model": model_type_name, "best_response": br_name, "cost": cost_name, "loss": loss_name,
		              "utility": utility_name}
	
	return model_spec