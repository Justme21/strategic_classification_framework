import torch

def hyper_grad_fuc(loss_val, loss_train, aux_params, params, learning_rate = .1, truncate_iter= 3):

	dloss_val_dparams = torch.autograd.grad(
		loss_val,
		params,
		retain_graph=True,
		allow_unused=True
	)

	dloss_train_dparams = torch.autograd.grad(
			loss_train,
			params,
			allow_unused=True,
			create_graph=True,
	)

	v2 = _approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params, learning_rate, truncate_iter)

	v3 = torch.autograd.grad(
		dloss_train_dparams,
		aux_params,
		grad_outputs=v2,
		allow_unused=True
	)


	return list(-g for g in v3)

def _approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params, learning_rate, truncate_iter):
	p = v = dloss_val_dparams

	for _ in range(truncate_iter):
		grad = torch.autograd.grad(
				dloss_train_dparams,
				params,
				grad_outputs=v,
				retain_graph=True,
				allow_unused=True
			)

		grad = [g * learning_rate for g in grad]

		v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
		p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

	return list(pp for pp in p)