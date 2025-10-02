import torch

from ..interfaces import BaseDataset, BaseModel
from .device_tools import get_device

def _get_ratio(y):
    # Assumes y in {-1, 1}
    return len(y[y==1]), len(y[y==-1])

def _evaluate_dataset(X,y):
    size = len(y)
    num_pos, num_neg = _get_ratio(y)
    return {"size": size, "pos:neg": f"{num_pos}:{num_neg} ({num_pos/size})"}

def _evaluate_model(model):
    details = {}
    if hasattr(model, "num_comps"):
        details["num_comps"] = model.num_comps
    if hasattr(model, "get_mixture_probs"):
        details["mixture_probs"] = model.get_mixture_probs().detach().tolist()

    details["weights"] = model.get_weights().detach().tolist()
    return details

def _calc_accuracy(label1, label2):
    return len(label1[label1==label2])*1.0/len(label1)

def _get_confusion_matrix(y, pred):
    pred = torch.where(pred==-1, 0, 1) 
    y = torch.where(y==-1, 0, 1)

    TP = (pred*y).sum().item()
    TN = ((1-pred)*(1-y)).sum().item()
    FP = (pred*(1-y)).sum().item()
    FN = ((1-pred)*y).sum().item()

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

def _evaluate_accuracy(model, X, y):
    with torch.enable_grad():
        # evaluate_model is done with grad disabled to freeze weights
        strat_X = model.best_response(X, model)

        pred_X = model.predict(X)
        pred_strat_X = model.predict(strat_X)

    clean_accuracy = _calc_accuracy(y, pred_X)
    strategic_accuracy = _calc_accuracy(y, pred_strat_X)

    clean_confusion = _get_confusion_matrix(y, pred_X)
    strat_confusion = _get_confusion_matrix(y, pred_strat_X)

    positive_gaming = strat_confusion["TP"] - clean_confusion["TP"] # Number of positive points that managed to correct classifier error
    negative_gaming = strat_confusion["FP"] - clean_confusion["FP"] # Number of successful gaming attempts

    return {"clean_accuracy": clean_accuracy,"strategic_accuracy": strategic_accuracy, "clean_confusion": clean_confusion, "strategic_confusion": strat_confusion,\
            "positive_gaming": positive_gaming, "negative_gaming": negative_gaming}

def evaluate_model(model:BaseModel, dataset:BaseDataset):
    results = {}

    X, y = dataset.get_all_vals()

    device = get_device()
    X, y = X.to(device), y.to(device)

    results['data stats'] = _evaluate_dataset(X, y)
    results['model details'] = _evaluate_model(model)
    results['accuracy'] = _evaluate_accuracy(model, X, y)

    return results

def validate_model(model:BaseModel, dataset:BaseDataset):

    X, y = dataset.get_all_vals()

    device = get_device()
    X, y = X.to(device), y.to(device)
    
    accuracy_metrics =  _evaluate_accuracy(model, X, y)
    return accuracy_metrics['clean_accuracy'], accuracy_metrics['strategic_accuracy']