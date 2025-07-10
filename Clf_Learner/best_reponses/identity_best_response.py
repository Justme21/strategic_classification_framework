from ..interfaces.base_best_response import BaseBestResponse

class IdentityResponse(BaseBestResponse):
    """Modelling when there is no Strategic Behaviour"""
    def __init__(self, utility, cost, **kwargs):
        pass

    def __call__(self, X, model):
        return X