from ..interfaces import BaseDataset, BaseModel

def _calc_accuracy(label1, label2):
    return len(label1[label1==label2])*1.0/len(label1)

def evaluate_model(model:BaseModel, dataset:BaseDataset):
    X, y = dataset.get_all_vals()
    strat_X = model.best_response.get_best_response(X, model)

    clean_accuracy = _calc_accuracy(y, model.predict(X))
    strategic_accuracy = _calc_accuracy(y, model.predict(strat_X))

    return {"clean_accuracy": clean_accuracy,"strategic_accuracy": strategic_accuracy}