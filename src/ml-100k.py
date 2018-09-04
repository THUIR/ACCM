import pandas as pd
from collections import Counter
import os
import numpy as np
import socket

np.random.seed(2017)
RAW_DATA = '../raw_data'
RATINGS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.data')
RATINGS = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
USERS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.user')
USERS = pd.read_csv(USERS_FILE, sep='|', header=None)
ITEMS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.item')
ITEMS = pd.read_csv(ITEMS_FILE, sep='|', header=None, encoding="ISO-8859-1")
OUT_DIR = '../dataset/'


def format_user_file(user_df):
    formatted = user_df[[0, 1, 2, 3]].copy()

    min_age, max_age = 15, 55
    formatted[1] = formatted[1].apply(lambda x: max_age if x > max_age else x)
    formatted[1] = formatted[1].apply(lambda x: min_age if x < min_age else x)
    formatted[1] = formatted[1].apply(lambda x: max_age / 5 if x >= max_age else min_age / 5 if x <= min_age else x / 5)
    # print Counter(formatted[1])
    formatted[1] = formatted[1].apply(lambda x: int(x - formatted[1].min()))
    formatted[2] = formatted[2].apply(lambda x: {'M': 0, 'F': 1}[x])
    occupation = dict(
        [(o.strip(), i) for i, o in enumerate(open(os.path.join(RAW_DATA, 'ml-100k/u.occupation'), 'r').readlines())])
    formatted[3] = formatted[3].apply(lambda x: occupation[x])
    formatted = formatted.fillna(-1)
    formatted.columns = ['uid', 'u_age', 'u_gender', 'u_occupation']
    # print formatted
    # print formatted.info()
    return formatted


def format_item_file(item_df):
    formatted = item_df.drop([1, 3, 4], axis=1).copy()
    formatted.columns = range(len(formatted.columns))
    formatted[1] = formatted[1].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else -1)

    min_year = 1989
    formatted[1] = formatted[1].apply(lambda x: min_year if 0 < x < min_year else x)
    formatted[1] = formatted[1].apply(lambda x: min_year + 1 if min_year < x < min_year + 4 else x)
    years = dict([(year, i) for i, year in enumerate(sorted(Counter(formatted[1]).keys()))])
    formatted[1] = formatted[1].apply(lambda x: years[x])
    formatted.columns = ['iid', 'i_year',
                         'i_Action', 'i_Adventure', 'i_Animation', "i_Children's", 'i_Comedy',
                         'i_Crime', 'i_Documentary ', 'i_Drama ', 'i_Fantasy ', 'i_Film-Noir ',
                         'i_Horror ', 'i_Musical ', 'i_Mystery ', 'i_Romance ', 'i_Sci-Fi ',
                         'i_Thriller ', 'i_War ', 'i_Western', 'i_Other']
    # print Counter(formatted[1])
    # print formatted
    # print formatted.info()
    return formatted


def format_rating(ratings, users, items):
    ratings = ratings.drop(3, axis=1).copy()
    ratings.columns = ['uid', 'iid', 'rating']
    ratings = pd.merge(ratings, users, on='uid', how='left')
    ratings = pd.merge(ratings, items, on='iid', how='left')
    # print ratings
    return ratings


def random_split_data():
    dir_name = 'ml-100k-r'
    if not os.path.exists(os.path.join(OUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUT_DIR, dir_name))
    users = format_user_file(USERS)
    users.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.users.csv'), index=False)
    items = format_item_file(ITEMS)
    items.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.items.csv'), index=False)
    all_data = format_rating(RATINGS, users, items)
    all_data.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.all.csv'), index=False)

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    train_size = int(len(all_data) * 0.8)
    validation_size = int(len(all_data) * 0.1)
    train_set = all_data[:train_size]
    validation_set = all_data[train_size:train_size + validation_size]
    test_set = all_data[train_size + validation_size:]
    # print train_set
    # print validation_set
    # print test_set
    train_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.train.csv'), index=False)
    validation_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.validation.csv'), index=False)
    test_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.test.csv'), index=False)


def split_cold_ui(item_cold_ratio=0.0, user_cold_ratio=0.0, vt_ratio=0.1, suffix=''):
    dir_name = 'ml-100k'
    if item_cold_ratio > 0:
        dir_name += '-i%d' % int(item_cold_ratio * 100)
    if user_cold_ratio > 0:
        dir_name += '-u%d' % int(user_cold_ratio * 100)
    dir_name += suffix
    if not os.path.exists(os.path.join(OUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUT_DIR, dir_name))

    users = format_user_file(USERS)
    users.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.users.csv'), index=False)
    items = format_item_file(ITEMS)
    items.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.items.csv'), index=False)
    all_data = format_rating(RATINGS, users, items)
    all_data.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.all.csv'), index=False)

    remain = all_data.copy()
    validation_size = int(len(remain) * vt_ratio)
    test_size = int(len(remain) * vt_ratio)
    validation_index = []
    test_index = []
    remain_index = []
    cold_item_index, cold_user_index = [], []

    if item_cold_ratio > 0:
        iid_list = remain.iid.unique().tolist()
        np.random.shuffle(iid_list)
        for iid in iid_list:
            iid_indexes = remain[remain.iid == iid].index.tolist()
            if len(cold_item_index) + len(iid_indexes) <= int(2 * validation_size * item_cold_ratio):
                cold_item_index.extend(iid_indexes)
                remain = remain.drop(iid_indexes)
            if len(cold_item_index) + len(iid_indexes) == int(2 * validation_size * item_cold_ratio):
                break
        cold_item_num = len(cold_item_index) / 2
        np.random.shuffle(cold_item_index)
        validation_index.extend(cold_item_index[:cold_item_num])
        test_index.extend(cold_item_index[cold_item_num:])

    if user_cold_ratio > 0:
        uid_list = remain.uid.unique().tolist()
        np.random.shuffle(uid_list)
        for uid in uid_list:
            uid_indexes = remain[remain.uid == uid].index.tolist()
            if len(cold_user_index) + len(uid_indexes) <= int(2 * validation_size * user_cold_ratio):
                cold_user_index.extend(uid_indexes)
                remain = remain.drop(uid_indexes)
            if len(cold_user_index) + len(uid_indexes) == int(2 * validation_size * user_cold_ratio):
                break
        cold_user_num = len(cold_user_index) / 2
        np.random.shuffle(cold_user_index)
        validation_index.extend(cold_user_index[:cold_user_num])
        test_index.extend(cold_user_index[cold_user_num:])

    remain_uid_index = []
    for uid, group in remain.groupby('uid'):
        remain_uid_index.extend(group.sample(1).index.tolist())
    remain_index.extend(remain_uid_index)
    remain = remain.drop(remain_uid_index)

    remain_iid_index = []
    for iid, group in remain.groupby('iid'):
        remain_iid_index.extend(group.sample(1).index.tolist())
    remain_index.extend(remain_iid_index)
    remain = remain.drop(remain_iid_index)

    sample_index = remain.sample(validation_size - len(validation_index)).index.tolist()
    validation_index.extend(sample_index)
    remain = remain.drop(sample_index)

    sample_index = remain.sample(test_size - len(test_index)).index.tolist()
    test_index.extend(sample_index)
    remain = remain.drop(sample_index)

    remain_index.extend(remain.index.tolist())

    validation_set = all_data.iloc[validation_index]
    test_set = all_data.iloc[test_index]
    train_set = all_data.iloc[remain_index]
    # print validation_set
    # print test_set
    # print train_set

    # train_set.sample(frac=1).to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.train.csv'), index=False)
    # validation_set.sample(frac=1).to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.validation.csv'),
    #                                      index=False)
    # test_set.sample(frac=1).to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.test.csv'), index=False)
    train_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.train.csv'), index=False)
    validation_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.validation.csv'),
                          index=False)
    test_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.test.csv'), index=False)
    print(len(set(validation_index + test_index + remain_index)))


def change_id(item_cold_ratio=0.0, user_cold_ratio=0.0, prefix='ml-100k-ci', suffix=''):
    if item_cold_ratio == 0.0 and user_cold_ratio == 0.0:
        split_cold_ui(0.0, 0.0, 0.1, suffix='-ci' + suffix)
        return
    dir_name = prefix
    if item_cold_ratio > 0:
        dir_name += '-i%d' % int(item_cold_ratio * 100)
    if user_cold_ratio > 0:
        dir_name += '-u%d' % int(user_cold_ratio * 100)
    dir_name += suffix
    if not os.path.exists(os.path.join(OUT_DIR, dir_name)):
        os.mkdir(os.path.join(OUT_DIR, dir_name))
    train_set = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.train.csv'))
    validation_set = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.validation.csv'))
    test_set = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.test.csv'))
    users = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.users.csv'))
    items = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.items.csv'))
    all_data = pd.read_csv(os.path.join(OUT_DIR, prefix + '/' + prefix + '.all.csv'))
    validation_cold_index, test_cold_index = [], []

    if item_cold_ratio > 0:
        i_columns = [c for c in validation_set.columns if c.startswith('i')]

        cold_size = int(len(validation_set) * item_cold_ratio)
        validation_set = validation_set.sample(frac=1.0)
        max_iid = items['iid'].max()
        validation_set['iid'][0:cold_size] = range(max_iid + 1, max_iid + cold_size + 1)
        # print validation_set
        new_items = validation_set[i_columns][0:cold_size]
        items = pd.concat([items, new_items])
        validation_cold_index.extend(validation_set.index[0:cold_size].tolist())
        # print validation_cold_index

        cold_size = int(len(test_set) * item_cold_ratio)
        test_set = test_set.sample(frac=1.0)
        max_iid = items['iid'].max()
        test_set['iid'][0:cold_size] = range(max_iid + 1, max_iid + cold_size + 1)
        new_items = test_set[i_columns][0:cold_size]
        items = pd.concat([items, new_items])
        test_cold_index.extend(test_set.index[0:cold_size].tolist())

    if user_cold_ratio > 0:
        u_columns = [c for c in validation_set.columns if c.startswith('u')]

        cold_size = int(len(validation_set) * user_cold_ratio)
        validation_set = validation_set.sample(frac=1.0)
        max_uid = users['uid'].max()
        validation_set['uid'][0:cold_size] = range(max_uid + 1, max_uid + cold_size + 1)
        new_users = validation_set[u_columns][0:cold_size]
        users = pd.concat([users, new_users])
        validation_cold_index.extend(validation_set.index[0:cold_size].tolist())

        cold_size = int(len(test_set) * user_cold_ratio)
        test_set = test_set.sample(frac=1.0)
        max_uid = users['uid'].max()
        test_set['uid'][0:cold_size] = range(max_uid + 1, max_uid + cold_size + 1)
        new_users = test_set[u_columns][0:cold_size]
        users = pd.concat([users, new_users])
        test_cold_index.extend(test_set.index[0:cold_size].tolist())

    validation_cold_index, test_cold_index = list(set(validation_cold_index)), list(set(test_cold_index))
    validation_set_cold = validation_set.loc[validation_cold_index]
    validation_set_warm = validation_set.drop(validation_cold_index)
    validation_set = pd.concat([validation_set_cold, validation_set_warm])
    test_set_cold = test_set.loc[test_cold_index]
    test_set_warm = test_set.drop(test_cold_index)
    test_set = pd.concat([test_set_cold, test_set_warm])

    train_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.train.csv'), index=False)
    validation_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.validation.csv'),
                          index=False)
    test_set.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.test.csv'), index=False)
    items.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.items.csv'), index=False)
    users.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.users.csv'), index=False)
    all_data.to_csv(os.path.join(OUT_DIR, dir_name + '/' + dir_name + '.all.csv'), index=False)
    # print len(set(validation_index + test_index + remain_index))


def main():
    random_split_data()
    change_id(item_cold_ratio=0.0, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.1, user_cold_ratio=0.1)
    # change_id(item_cold_ratio=0.2, user_cold_ratio=0.2)
    change_id(item_cold_ratio=0.3, user_cold_ratio=0.3)
    # change_id(item_cold_ratio=0.4, user_cold_ratio=0.4)
    # change_id(item_cold_ratio=0.5, user_cold_ratio=0.5)
    #
    # change_id(item_cold_ratio=0.1, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.2, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.3, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.4, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.5, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.6, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.7, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.8, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=0.9, user_cold_ratio=0.0)
    # change_id(item_cold_ratio=1.0, user_cold_ratio=0.0)
    #
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.1)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.2)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.3)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.4)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.5)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.6)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.7)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.8)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=0.9)
    # change_id(item_cold_ratio=0.0, user_cold_ratio=1.0)


if __name__ == '__main__':
    main()
