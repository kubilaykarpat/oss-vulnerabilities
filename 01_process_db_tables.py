import re
import time
from collections import defaultdict

import pandas as pd
import pymysql

import vul_common

entity_prefix = 'FUNCTIONS'
repo_id_regex = re.compile('^FUNCTIONS_(\\d+)_.*$')
sub_repo_name_regex = re.compile('^FUNCTIONS_(?:\\d+)_(.*)$')

main_start = time.time()
db_connection = pymysql.connect(host="localhost",  # your host, usually localhost
                                user="root",  # your username
                                # passwd="localhost",  # your password
                                db="security")  # name of the data base
print("------> Tasks started")
with db_connection:
    dict_cursor = db_connection.cursor(pymysql.cursors.DictCursor)
    dict_cursor.execute("SELECT * FROM REPOSITORIES_SAMPLE")
    projects = dict_cursor.fetchall()
    projects = {project['R_ID']: project['PROJECT'] for project in projects}

    cursor = db_connection.cursor()
    cursor.execute("SHOW TABLES LIKE '{}%'".format(entity_prefix))
    tables = [item[0] for item in cursor.fetchall()]

    tables_of_repos = defaultdict(list)
    for table_name in tables:
        project_id = repo_id_regex.search(table_name).group(1)
        tables_of_repos[project_id].append(table_name)

    for project_id, tables_of_repo in tables_of_repos.items():
        project_start = time.time()
        df_of_repo = None
        name_of_project = projects[int(project_id)]

        if name_of_project not in vul_common.projects_names:
            continue

        for table_name in tables_of_repo:
            sub_project_start = time.time()
            name_of_sub_project = sub_repo_name_regex.search(table_name).group(1)
            query = "SELECT * FROM {}".format(table_name)
            df = pd.read_sql(query, db_connection)
            df['oss_sub_project'] = name_of_sub_project
            if df_of_repo is None:
                df_of_repo = df
            else:
                df_of_repo.append(df)
            sub_project_time = time.time() - main_start
            print("----> {}/{} finished in {}".format(name_of_project, name_of_sub_project, sub_project_time))
        df_of_repo['oss_project'] = name_of_project
        df_of_repo.to_csv(vul_common.table_csv_filename(name_of_project))
        print("LOL")

        project_time = time.time() - main_start
        print("--> {} finished in {}".format(name_of_project, project_time))

main_time = time.time() - main_start
print("------> All tasks finished in {}".format(main_time))
