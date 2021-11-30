import os
import numpy as np
import pandas as pd
from pandas.core.arrays import base


def generate_files_for_plots():
    src_file = pd.read_csv(os.path.join("Results", "Results_summary.csv"))
    root_dst_pth = os.path.join("Results", "Plots")

    if not os.path.exists(root_dst_pth):
        os.makedirs(root_dst_pth)

    base_classifiers = src_file["cls"].unique()
    red_algorithms = src_file["red_alg"].unique()

    for base_classifier in base_classifiers:
        current_classifier_path = os.path.join(root_dst_pth, base_classifier)
        if not os.path.exists(current_classifier_path):
            os.makedirs(current_classifier_path)

        tmp = (
            src_file.loc[(src_file["cls"] == base_classifier)]
            .sort_values(by=["HL", "Size"], ascending=(True, True))
            .to_csv(
                os.path.join(current_classifier_path, "ND.csv"),
                index=False,
                float_format="%.3f",
            )
        )
        # Splitting by reduction algorithm
        for red_algorithm in red_algorithms:
            current_excerpt = src_file.loc[
                (src_file["cls"] == base_classifier)
                & (src_file["red_alg"] == red_algorithm)
            ]

            current_excerpt.to_csv(
                os.path.join(current_classifier_path, "{}.csv".format(red_algorithm)),
                index=False,
                float_format="%.3f",
            )
    return


if __name__ == "__main__":
    generate_files_for_plots()
