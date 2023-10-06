import os
import pickle

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    print("Loading data...")
    folder = "svm"
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_file = os.path.join(folder, "metrics.txt")
    with open("allArffs.pckl", "rb") as file:
        arffs = pickle.load(file)
    print("Data loaded.")

    print("Preparing data...")
    labels = list(map(lambda x: x["@@class@@"] == b'1', arffs))
    data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.33, random_state=42)

    X_train = pd.concat(train_data).to_numpy()
    X_test = pd.concat(test_data).to_numpy()
    y_train = pd.concat(train_labels).to_numpy()
    y_test = pd.concat(test_labels).to_numpy()
    print("Data prepared, training SVM...")
    svc: SVC = SVC(verbose=True)
    svc.fit(X_train, y_train)
    with open(os.path.join(folder, "RFC.pckl"), "wb") as file:
        pickle.dump(svc, file)
    print("SVM trained, calculating metrics...")
    y_pred = svc.predict(X_test)
    with open(output_file, "w") as file:
        print(f"    accuracy: {accuracy_score(y_test, y_pred)}", file=file)
        print(f"    precision: {precision_score(y_test, y_pred)}", file=file)
        print(f"    recall: {recall_score(y_test, y_pred)}", file=file)
        print(f"    f1_score: {f1_score(y_test, y_pred)}", file=file)
        print(f"SVM trained succesfully with f1: {f1_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()
