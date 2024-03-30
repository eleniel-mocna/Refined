from sklearn.metrics import f1_score


def get_best_cutoff(data, labels, random_forest):
    y_pred = random_forest.predict_proba(data)
    best_cutoff = 0
    best_f1 = 0
    for i in range(1, 100):
        cutoff = i / 100
        f1 = f1_score(labels, y_pred[:, 1] > cutoff)
        if f1 > best_f1:
            best_cutoff = cutoff
            best_f1 = f1
    print(f"RFC trained with f1: {best_f1}")
    return best_cutoff
