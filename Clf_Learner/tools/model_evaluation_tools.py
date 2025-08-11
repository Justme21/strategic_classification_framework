import torch

from ..interfaces import BaseDataset, BaseModel

def _get_ratio(y):
    # Assumes y in {-1, 1}
    return len(y[y==1]), len(y[y==-1])

def _evaluate_dataset(X,y):
    size = len(y)
    num_pos, num_neg = _get_ratio(y)
    return {"size": size, "pos:neg": f"{num_pos}:{num_neg} ({num_pos/size})"}

def _calc_accuracy(label1, label2):
    return len(label1[label1==label2])*1.0/len(label1)

#def _get_confusion_matrix(model, X, y):
def _get_confusion_matrix(y, pred):
    #pred = model.predict(X)

    pred = torch.where(pred==-1, 0, 1) 
    y = torch.where(y==-1, 0, 1)

    TP = (pred*y).sum().item()
    TN = ((1-pred)*(1-y)).sum().item()
    FP = (pred*(1-y)).sum().item()
    FN = ((1-pred)*y).sum().item()

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

def _evaluate_accuracy(model, X, y, strat_X):
    clean_accuracy = _calc_accuracy(y, model.predict(X))
    strategic_accuracy = _calc_accuracy(y, model.predict(strat_X))

    clean_confusion = _get_confusion_matrix(model, X, y)
    strat_confusion = _get_confusion_matrix(model, strat_X, y)

    return {"clean_accuracy": clean_accuracy,"strategic_accuracy": strategic_accuracy, "clean_confusion": clean_confusion, "strategic_confusion": strat_confusion}

def evaluate_model(model:BaseModel, dataset:BaseDataset):
    results = {}

    X, y = dataset.get_all_vals()
    with torch.enable_grad():
        # evaluate_model is done with grad disabled to freeze weights
        strat_X = model.best_response(X, model)

    results['data stats'] = _evaluate_dataset(X, y)
    results['accuracy'] = _evaluate_accuracy(model, y, X, strat_X)

    return results
    