import os

selected_metric = 'FUNCTIONS'  # 'FUNCTIONS' or 'CLASSES' or 'FILES'
random_state = 1994

if not os.path.exists('out'):
    os.makedirs('out')

if not os.path.exists('out/table'):
    os.makedirs('out/table')

if not os.path.exists('out/fold'):
    os.makedirs('out/fold')

if not os.path.exists('out/evaluations'):
    os.makedirs('out/evaluations')

projects_names = ['Apache', 'Glibc', 'Kernel Linux', 'Mozilla', 'Xen']


def table_csv_filename(project_name):
    return "out/table/table_{}_{}.csv".format(selected_metric, project_name)


def metrics_train_fold_csv_filename(project_name, fold_no):
    return "out/fold/fold_{}_train_{}_{}.csv".format(fold_no, selected_metric, project_name)


def metrics_test_fold_csv_filename(project_name, fold_no):
    return "out/fold/fold_{}_test_{}_{}.csv".format(fold_no, selected_metric, project_name)


def merge_dfs(dfs):
    merged = None
    for df in dfs:
        if merged:
            merged.append(df)
        else:
            merged = df.copy()

    return merged


number_of_folds = 10
test_vulnerability_sampling_ratio = 0.1
features_cols = ["AltCountLineCode", "CountInput", "CountLineBlank", "CountLineCodeDecl", "CountLineComment",
                 "CountLinePreprocessor", "CountPath", "CountStmt", "CountStmtEmpty", "Cyclomatic",
                 "CyclomaticStrict", "Knots", "MinEssentialKnots", "RatioCommentToCode", "AltCountLineComment",
                 "CountLine", "CountLineCode", "CountLineCodeExe", "CountLineInactive", "CountOutput",
                 "CountSemicolon", "CountStmtDecl", "CountStmtExe", "CyclomaticModified", "Essential",
                 "MaxEssentialKnots",
                 "MaxNesting"]
label_col = "Vulnerable"
