def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    truepos = (prediction * ground_truth).sum()
    falsepos = (prediction[(ground_truth != 0)[0]]).sum()
    falseneg = (ground_truth[(prediction != 0)[0]]).sum()
    trueneg = ground_truth.shape[0] - truepos - falseneg - falsepos

    accuracy = (truepos + trueneg) / ground_truth.shape[0]

    precision = 0 if (truepos + falsepos) == 0 else truepos / (truepos + falsepos)

    recall = 0 if (truepos + falseneg) == 0 else truepos / (truepos + falseneg)

    f1 = 0 if (recall + precision) == 0 else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return sum(prediction == ground_truth) / prediction.shape[0]