import time
from collections import defaultdict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import vul_common


def train_and_measure(train_features, train_labels, test_features, test_labels):
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_jobs=2, random_state=vul_common.random_state, n_estimators=50)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    clf.fit(train_features, train_labels)

    score = clf.score(test_features, test_labels)
    print(score)


main_start = time.time()
project_train_folds = defaultdict(list)  # {Project Name, [(Features#DataFrame), Labels#Series)}>
project_test_folds = defaultdict(list)  # {Project Name, [(Features#DataFrame), Labels#Series)}>

for project_name in vul_common.projects_names:
    for fold_num in range(vul_common.number_of_folds):
        train_file_name = vul_common.metrics_train_fold_csv_filename(project_name, fold_num)
        fold_train_df = pd.read_csv(train_file_name)
        fold_train_features = fold_train_df[vul_common.features_cols]
        fold_train_labels = fold_train_df[vul_common.label_col]
        project_train_folds[project_name].append((fold_train_features, fold_train_labels))

        test_file_name = vul_common.metrics_test_fold_csv_filename(project_name, fold_num)
        fold_test_df = pd.read_csv(test_file_name)
        fold_test_features = fold_test_df[vul_common.features_cols]
        fold_test_labels = fold_test_df[vul_common.label_col]
        project_test_folds[project_name].append((fold_test_features, fold_test_labels))

# M1: General Model

general_train_folds = []  # [(Features#DataFrame: Labels#Series]
general_test_folds = []  # [(Features#DataFrame: Labels#Series]

for fold_no in range(vul_common.number_of_folds):
    # [(Features(DataFrame), Labels(Series))]
    train_data_of_fold = [folds_of_project[fold_no] for folds_of_project in project_train_folds.values()]
    merged_train_features_of_fold = pd.concat([project_fold[0] for project_fold in train_data_of_fold])
    merged_train_labels_of_fold = pd.concat([project_fold[1] for project_fold in train_data_of_fold])
    general_train_folds.append((merged_train_features_of_fold, merged_train_labels_of_fold))

    # [(Features(DataFrame), Labels(Series))]
    test_data_of_fold = [folds_of_project[fold_no] for folds_of_project in project_test_folds.values()]
    merged_test_features_of_fold = pd.concat([project_fold[0] for project_fold in test_data_of_fold])
    merged_test_labels_of_fold = pd.concat([project_fold[1] for project_fold in test_data_of_fold])
    general_test_folds.append((merged_test_features_of_fold, merged_test_labels_of_fold))

print("----> General model evaluation started")
for fold_no in range(vul_common.number_of_folds):
    train_and_measure(general_train_folds[fold_no][0], general_train_folds[fold_no][1],
                      general_test_folds[fold_no][0], general_test_folds[fold_no][1])
print("----> General model evaluation finished")


for project_name in vul_common.projects_names:
    print("----> Within project model evaluation started for {}".format(project_name))
    #all_train_folds_of_other_projects = [fold_]


    print("----> Within project model evaluation started for {}".format(project_name))

main_time = time.time() - main_start
print("------> All tasks finished in {}".format(main_time))
