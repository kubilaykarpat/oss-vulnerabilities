import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import vul_common

rf_n_jobs = 2
rf_n_estimators = 50


def calculate_tp_fp_tn_fn(confusion_matrix):
    return confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][1], confusion_matrix[1][0]


def calculate_tp_fp_tn_fn_for_multi_class(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = confusion_matrix.sum() - (fp + fn + tp)
    return tp, fp, tn, fn


def calculate_informedness(fp, fn, tp, tn):
    return (tp / (tp + fn)) - (fp / (tn + fp))


def calculate_markedness(fp, fn, tp, tn):
    return (tp / (tp + fp)) - (fn / (fn + tn))


def draw_roc_graph(experiment_out_path, experiment_name, project_name, roc_metrics_for_project):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    plt.clf()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    count = 0
    for fpr, tpr in roc_metrics_for_project:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))
        count += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("{}/roc_{}_{}.png".format(experiment_out_path, experiment_name, project_name))

def train_and_measure(train_samples, test_samples):
    function_time = time.time()

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_jobs=rf_n_jobs, random_state=vul_common.random_state, n_estimators=rf_n_estimators)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    start = time.time()
    clf.fit(train_samples[vul_common.features_cols], train_samples[vul_common.label_col])
    print("Training took {} seconds".format(time.time() - start))

    scores = {}
    start = time.time()
    predicted = clf.predict(test_samples[vul_common.features_cols])
    print("Predict labels took {} seconds".format(time.time() - start))
    actual = test_samples[vul_common.label_col]
    scores['accuracy'] = metrics.accuracy_score(actual, predicted)
    scores['recall'] = metrics.recall_score(actual, predicted)
    scores['precision'] = metrics.precision_score(actual, predicted)
    scores['f_measure'] = metrics.f1_score(actual, predicted)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    tp, fp, tn, fn = calculate_tp_fp_tn_fn(confusion_matrix)
    scores['informedness'] = calculate_informedness(fp, fn, tp, tn)
    scores['markedness'] = calculate_markedness(fp, fn, tp, tn)

    start = time.time()
    probas_ = clf.predict_proba(test_samples[vul_common.features_cols])
    print("Predict probabilities took {} seconds".format(time.time() - start))

    fpr, tpr, thresholds = metrics.roc_curve(actual, probas_[:, 1])
    print("->Train and measure took {} seconds".format(time.time() - function_time))

    return scores, fpr, tpr, thresholds


def create_result_line(project_name, fold_no, score):
    return {'project': project_name, 'fold_no': fold_no, **score}


main_start = time.time()
project_train_folds = defaultdict(list)  # { Project Name, List of Folds#[DataFrame] }>
project_test_folds = defaultdict(list)  # { Project Name, List of Folds#[DataFrame] }>

for pivot_project in vul_common.projects_names:
    for fold_num in range(vul_common.number_of_folds):
        train_file_name = vul_common.metrics_train_fold_csv_filename(pivot_project, fold_num)
        fold_train_df = pd.read_csv(train_file_name)
        project_train_folds[pivot_project].append(fold_train_df)

        test_file_name = vul_common.metrics_test_fold_csv_filename(pivot_project, fold_num)
        fold_test_df = pd.read_csv(test_file_name)
        project_test_folds[pivot_project].append(fold_test_df)

experiment_out_path = "out/evaluations/{}".format(time.strftime("%Y%m%dT%H%MZ"))
os.makedirs(experiment_out_path)

parameters = {
    'projects': vul_common.projects_names,
    'features': vul_common.features_cols,
    'labels': vul_common.label_col,
    'number_of_folds': vul_common.number_of_folds,
    'rf_n_jobs': rf_n_jobs,
    'rf_n_estimators': rf_n_estimators,
    'random_state': vul_common.random_state
}

with open(experiment_out_path + '/parameters.json', 'w') as parameters_file:
    json.dump(parameters, parameters_file, indent=4, sort_keys=True)

cross_project_results = []
for pivot_project in vul_common.projects_names:
    print("----> Cross project model evaluation started for {}".format(pivot_project))
    project_start = time.time()

    roc_metrics_for_project = []  # [(fpr, tpr)]
    for fold_no in range(vul_common.number_of_folds):
        print("--> Cross project fold-{} started for {}".format(fold_no, pivot_project))
        fold_start = time.time()

        # Let's first gather all folds for projects except pivot project
        train_folds = [fold for project_name, folds_of_project in project_train_folds.items()
                       if project_name != pivot_project
                       for fold in folds_of_project]

        # Then add all folds of within project except selected fold_no
        train_folds_of_within_project = [fold for index, fold in enumerate(project_train_folds[pivot_project]) if
                                         index != fold_no]
        train_folds = train_folds + train_folds_of_within_project

        # Then pick the test for fold for pivot project
        test_fold = project_test_folds[pivot_project][fold_no]
        score, fpr, tpr, _ = train_and_measure(pd.concat(train_folds), test_fold)
        roc_metrics_for_project.append((fpr, tpr))
        cross_project_results.append(create_result_line(pivot_project, fold_no, score))

        fold_time = time.time() - fold_start
        print("--> Cross project fold-{} for {} finished in {} seconds".format(fold_no, pivot_project, fold_time))

    draw_roc_graph(experiment_out_path, "cross_project", pivot_project, roc_metrics_for_project)

    project_time = time.time() - project_start
    print("----> Cross project model evaluation for {} finished in {} seconds".format(pivot_project, project_time))

evaluations = pd.DataFrame.from_records(cross_project_results)
evaluations.to_csv(experiment_out_path + '/cross_projects_results.csv')

within_project_results = []
for pivot_project in vul_common.projects_names:
    print("----> Within project model evaluation started for {}".format(pivot_project))
    project_start = time.time()
    roc_metrics_for_project = []  # [(fpr, tpr)]
    for fold_no in range(vul_common.number_of_folds):
        print("--> Within project fold-{} started for {}".format(fold_no, pivot_project))
        fold_start = time.time()

        # Select train folds for this project except for the fold_no
        train_folds_of_within_project = [fold for index, fold in enumerate(project_train_folds[pivot_project]) if
                                         index != fold_no]

        # Select test folds for this project except for the fold_no
        test_fold = project_test_folds[pivot_project][fold_no]

        score, fpr, tpr, _ = train_and_measure(pd.concat(train_folds_of_within_project), test_fold)
        within_project_results.append(create_result_line(pivot_project, fold_no, score))
        roc_metrics_for_project.append((fpr, tpr))

        fold_time = time.time() - fold_start
        print("--> Within project fold-{} for {} finished in {} seconds".format(fold_no, pivot_project, fold_time))

    draw_roc_graph(experiment_out_path, "within_project", pivot_project, roc_metrics_for_project)

    project_time = time.time() - project_start
    print("----> Within project model evaluation for {} finished in {} seconds".format(pivot_project, project_time))

evaluations = pd.DataFrame.from_records(within_project_results)
evaluations.to_csv(experiment_out_path + '/within_projects_results.csv')

main_time = time.time() - main_start
print("------> All tasks finished in {} seconds".format(main_time))
