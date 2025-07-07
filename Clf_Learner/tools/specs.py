SPEC_DICT = {
    #"<recipe_name>": {"model": <model_name>, "best_response": <best_response_name>, "cost": <cost_name>, "loss": <loss_name>, "utility": <utility_name>}
    "naive_SSVM": {"model": "linear", "best_response": "linear", "cost": None, "loss": "naive_strategic_hinge", "utility": None}, # Strategic Classification Made Practical 
     "SSVM": {"model": "linear", "best_response": "linear", "cost": None, "loss": "strategic_hinge", "utility": None}, #Strategic Classification Made Practical
}