import numpy as np

def compute_metrics(eval_preds):

    metric = dict()

    preds = eval_preds.predictions
    labels = eval_preds.label_ids

    preds = np.argmax(preds, axis=1)
    labels = labels

    metric['accuracy'] = (preds == labels).mean()

    return metric
