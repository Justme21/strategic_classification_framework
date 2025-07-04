from ..interfaces.base_best_response import BaseBestResponse

class IdentityResponse(BaseBestResponse):
    """Modelling when there is no Strategic Behaviour"""
    def __init__(self, utility, cost):
        pass

    def get_best_response(self, X, model):
        return X