import time
from collections import defaultdict
import vul_common

import pandas as pd


main_start = time.time()
project_feature_folds = defaultdict(list)  # <Project Name, [(Train Folds, Test Folds)]>
project_label_folds = defaultdict(list)  # <Project Name, [(Train Folds, Test Folds)]>
file = "out/test.csv"
pd.read_csv(file)
for project_name in vul_common.projects_names:
    for fold_num in range(vul_common.number_of_folds):
        file_name = vul_common.metrics_test_fold_csv_filename(project_name, fold_num)
        fold_df = pd.read_csv(file_name)
        fold_features = fold_df[vul_common.features_cols]
        fold_labels = fold_df(vul_common.label_col)
        project_feature_folds[project_name].append((fold_features, fold_labels))

main_time = time.time() - main_start
print("------> All tasks finished in {}".format(main_time))
