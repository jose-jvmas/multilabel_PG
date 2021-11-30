import os
from pathlib import Path
import numpy as np
import pandas as pd
from ALL import ALL
from MRHC import MRHC
from MChen import MChen
from MRSP3 import MRSP3
from MRSP2 import MRSP2
from MRSP1 import MRSP1
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.dataset import available_data_sets
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
import scipy.sparse

from load_mtg_jamendo_dataset import load_mtg_jamendo_dataset

results_path_root = "Results"
reduction_path = "Reduction"
scores_path = "Scores"

jamendo_root = Path("/scratch/palonso/data/mtg-jamendo-dataset-mood/embeddings/")

dimension_map = {
    "effnet": 200,
    "musicnn": 200,
    "openl3": 512,
    "vggish": 128,
    "yamnet": 1024,
}


def loadCorpus(corpus_name):
    if "jamendo" in corpus_name:
        embedding_type = corpus_name.split("-")[-1]
        jamendo_path = jamendo_root / f"embeddings-{embedding_type}"
        dimensions = dimension_map[embedding_type]

        X, y = load_mtg_jamendo_dataset(
            jamendo_path,
            dimensions,
            merge_validation=False,  # For now discard the validation set, so that we can fairly compare to the Challenge models, but we could leverage that data for training.
        )

        return X["train"], y["train"], X["test"], y["test"]

    else:
        X_train, y_train, _, _ = load_dataset(corpus_name, "train")
        X_test, y_test, _, _ = load_dataset(corpus_name, "test")

    return X_train, y_train, X_test, y_test


def experiments():

    # Reduction path:
    if not os.path.exists(os.path.join(results_path_root, reduction_path)):
        os.makedirs(os.path.join(results_path_root, reduction_path))

    # Selected corpora:
    # corpora = [
    #     "bibtex",
    #     "birds",
    #     "Corel5k",
    #     "emotions",
    #     "enron",
    #     "genbase",
    #     "medical",
    #     "rcv1subset1",
    #     "rcv1subset2",
    #     "rcv1subset3",
    #     "rcv1subset4",
    #     "scene",
    #     "yeast",
    # ]

    corpora = [
        "bibtex",
        "birds",
        "Corel5k",
        "emotions",
        "genbase",
        "medical",
        "rcv1subset1",
        "rcv1subset2",
        "rcv1subset3",
        "rcv1subset4",
        "scene",
        "yeast",
        # MTG-Jamendo Mood and Theme Recognition tasks -> https://multimediaeval.github.io/2021-Emotion-and-Theme-Recognition-in-Music-Task/
        # Train version based on different embedding types
        "jamendo-effnet",
        "jamendo-musicnn",
        "jamendo-openl3",
        "jamendo-vggish",
        "jamendo-yamnet",
    ]

    # Reduction algorithms:
    red_algs = ["ALL", "MRSP1", "MRSP2", "MRSP3", "MRHC", "MChen"]

    # Params dict:
    red_algos_param = {
        "ALL": [1],
        "MRHC": [1],
        "MRSP3": [1],
        "MRSP1": [10, 30, 50, 70, 90],
        "MRSP2": [10, 30, 50, 70, 90],
        "MChen": [10, 30, 50, 70, 90],
    }

    # Classifiers:
    classifiers = ["LabelPowerset", "MLkNN", "BRkNNaClassifier", "BRkNNbClassifier"]

    # Classifier params:
    classifiers_param = {
        "LabelPowerset": [1, 3, 5, 7],
        "MLkNN": [1, 3, 5, 7],
        "BRkNNaClassifier": [1, 3, 5, 7],
        "BRkNNbClassifier": [1, 3, 5, 7],
    }

    out_file = pd.DataFrame(
        columns=[
            "cls",
            "cls_params",
            "red_alg",
            "red_alg_params",
            "corpus",
            "HL",
            "Size",
        ]
    )

    res_line = dict()

    for single_corpus in corpora:

        # Dst folder:
        corpus_dst_path = os.path.join(results_path_root, reduction_path, single_corpus)
        res_line["corpus"] = [single_corpus]
        if not os.path.exists(corpus_dst_path):
            os.makedirs(corpus_dst_path)

        # Loading corpus:
        X_train, y_train, X_test, y_test = loadCorpus(single_corpus)
        print("-" * 80)
        print("CORPUS {}".format(single_corpus))
        for single_red in red_algs:

            print("- Reduction {}".format(single_red))
            res_line["red_alg"] = [single_red]

            red = eval(single_red + "()")

            for red_parameter in red_algos_param[single_red]:
                res_line["red_alg_params"] = [str(red_parameter)]
                print("-- Reduction parameter {}".format(red_parameter))

                X_dst_file = os.path.join(
                    corpus_dst_path, "X_" + red.getFileName(red_parameter) + ".csv"
                )
                y_dst_file = os.path.join(
                    corpus_dst_path, "y_" + red.getFileName(red_parameter) + ".csv"
                )

                if os.path.isfile(X_dst_file):
                    X_red = np.array(pd.read_csv(X_dst_file, sep=",", header=None))
                    y_red = np.array(pd.read_csv(y_dst_file, sep=",", header=None))
                else:
                    if type(X_train) is scipy.sparse:
                        X_train = X_train.toarray()
                    if type(y_train) is scipy.sparse:
                        y_train = y_train.toarray()

                    X_red, y_red = red.reduceSet(
                        X=X_train, y=y_train, params=red_parameter
                    )

                    pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None)
                    pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None)

                for single_classifier in classifiers:
                    print("--- Classifier {}".format(single_classifier))

                    res_line["cls"] = [single_classifier]

                    for classifier_parameters in classifiers_param[single_classifier]:
                        print("---- Classifier param {}".format(classifier_parameters))

                        res_line["cls_params"] = [str(classifier_parameters)]

                        if single_classifier == "LabelPowerset":
                            kNN = KNeighborsClassifier(
                                n_neighbors=classifier_parameters
                            )
                            cls = LabelPowerset(
                                classifier=kNN, require_dense=[False, False]
                            )
                        else:
                            cls = eval(
                                single_classifier
                                + "(k="
                                + str(classifier_parameters)
                                + ")"
                            )

                        cls.fit(X_red, y_red)
                        y_pred = cls.predict(X_test)

                        res_line["HL"] = [hamming_loss(y_test, y_pred)]
                        res_line["Size"] = [100 * X_red.shape[0] / X_train.shape[0]]

                        print("----- DONE!")

                        out_file = out_file.append(
                            pd.DataFrame(res_line), ignore_index=False
                        )
                        out_file.to_csv(
                            os.path.join(results_path_root, "Results_plain.csv"),
                            index=False,
                        )

    # out_file = out_file.sort_values(by=['cls', 'cls_params', 'red_alg', 'red_alg_params', 'corpus'], ascending=[True, True, True, True, True])
    # out_file.to_csv(os.path.join(results_path_root, 'Results_plain.csv'))
    out_file.groupby(
        ["cls", "cls_params", "red_alg", "red_alg_params"]
    ).mean().reset_index().to_csv(
        os.path.join(results_path_root, "Results_summary.csv"), index=False
    )

    return


def getDataStats():
    # print(set([x[0] for x in available_data_sets().keys()]))
    print("-" * 80)
    print("Name, Train size, Test size, Features, Possible tags, Cardinality, Density")
    for single_key in sorted(list(set([u[0] for u in available_data_sets().keys()]))):
        X_train, y_train, _, _ = load_dataset(single_key, "train")
        X_test, y_test, _, _ = load_dataset(single_key, "test")

        cardinality = (
            np.count_nonzero(y_train.toarray()) + np.count_nonzero(y_test.toarray())
        ) / (y_train.shape[0] + y_test.shape[0])

        print(
            "{},{},{},{},{},{},{}".format(
                single_key,
                X_train.shape[0],
                X_test.shape[0],
                X_train.shape[1],
                y_train.shape[1],
                cardinality,
                cardinality / y_train.shape[1],
            )
        )

    return


if __name__ == "__main__":

    """Corpora stats"""
    # getDataStats()

    """Running experiments"""
    experiments()

    # https://www.javatips.net/api/Keel3.0-master/src/keel/Algorithms/Instance_Generation/Chen/ChenGenerator.java
