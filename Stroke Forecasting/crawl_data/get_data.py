import pandas as pd
import pymysql
import configparser
import argparse
import os
from database_api.db_download import db_download, db_api

cfg_parser = configparser.ConfigParser()
cfg_parser.read('db.cfg')
db_config = cfg_parser['CoachAI']
dl = db_download(db_config['host'],
                int(db_config['port']),
                db_config['user'],
                db_config['password'],
                db_config['database'])
da = db_api(db_config['host'],
            int(db_config['port']),
            db_config['user'],
            db_config['password'],
            db_config['database'])


def get_match():
    all_match = dl.download_match_csv('match.csv', True)

    # filter matches
    df = pd.read_csv('match.csv')
    checked_list = da.get_graph_check_list()
    df = df[df['id'].isin(checked_list)]
    df.to_csv('match.csv', index=False)


def get_set():
    checked_list = check_match()
    match_list = pd.read_csv('match.csv')
    match_id_list = match_list['id'].values
    match_name_list = match_list['video'].values

    success_count = 0
    for id, name in zip(match_id_list, match_name_list):
        if id in checked_list:
            success_count += 1
            path = 'data/' + name
            os.makedirs(path, exist_ok=True)
            dl.download_set_csv(id, path, True)
        else:
            pass
    print("Total crawling matches: {}".format(success_count))


def check_match():
    checked_list = da.get_graph_check_list()
    print(len(checked_list), checked_list)
    return checked_list


def get_homography():
    dl.download_homography_matrix_csv('homography.csv')

    # filter matches
    df = pd.read_csv('homography.csv')
    checked_list = da.get_graph_check_list()
    df = df[df['id'].isin(checked_list)]
    df.to_csv('homography.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['get_match', 'get_set', 'get_homography', 'check_match'],
                        required=True,
                        help="operation mode")

    opt = parser.parse_args()

    if opt.mode == 'get_match':
        get_match()
    elif opt.mode == 'get_set':
        get_set()
    elif opt.mode == 'get_homography':
        get_homography()
    elif opt.mode == 'check_match':
        check_match()