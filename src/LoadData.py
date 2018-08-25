import os
import pandas as pd
import numpy as np


class LoadData(object):
    def __init__(self, path, dataset, load_data=True, sep='\t', label='label', append_id=True, include_id=False,
                 seqs_features=None, seqs_sep=',', seqs_expand=True):
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + '.train.csv')
        self.validation_file = os.path.join(self.path, dataset + '.validation.csv')
        self.test_file = os.path.join(self.path, dataset + '.test.csv')
        self.max_file = os.path.join(self.path, dataset + '.max.csv')
        self.include_id = include_id
        self.append_id = append_id
        self.label = label
        self.sep = sep
        self.seqs_features = [] if seqs_features is None else seqs_features
        self.seqs_sep = seqs_sep
        self.seqs_expand = seqs_expand
        self.word_embeddings = None
        self.load_data = load_data
        self._load_data()
        self._load_max()
        self._load_words()
        self._select_features()

        self.train_data = None if self.train_df is None else self.format_data(self.train_df)
        print("# of train:", len(self.train_data['Y']))
        self.validation_data = None if self.validation_df is None else self.format_data(self.validation_df)
        print("# of validation:", len(self.validation_data['Y']))
        self.test_data = None if self.test_df is None else self.format_data(self.test_df)
        print("# of test:", len(self.test_data['Y']))

    def _load_data(self):
        self.train_df, self.validation_df, self.test_df = None, None, None
        if os.path.exists(self.train_file) and self.load_data:
            print("load train csv...", end='')
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            print("done")
        if os.path.exists(self.validation_file) and self.load_data:
            print("load validation csv...", end='')
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            print("done")
        if os.path.exists(self.test_file) and self.load_data:
            print("load test csv...", end='')
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            print("done")

    def _load_max(self):
        max_series = None
        if not os.path.exists(self.max_file):
            for df in [self.train_df, self.validation_df, self.test_df]:
                if df is not None:
                    df_max = df.max()
                    for seq in self.seqs_features:
                        seqs = df[seq].str.split(',')
                        seqs = seqs.apply(lambda x: len(x))
                        df_max[seq] = seqs.max() + 1
                    max_series = df_max if max_series is None else np.maximum(max_series, df_max)
            max_series.to_csv(self.max_file, sep=self.sep)
        else:
            max_series = pd.read_csv(self.max_file, sep=self.sep, header=None)
            max_series = max_series.set_index(0, drop=True).transpose()
            max_series = max_series.loc[1]
        self.column_max = max_series

    def _load_words(self):
        self.dictionary = {}
        for seq_feature in self.seqs_features:
            print('max length of %s: %d' % (seq_feature, self.column_max[seq_feature]))
            word_file = os.path.join(self.path, self.dataset + '.%s_word.csv' % seq_feature)
            words = pd.read_csv(word_file, sep='\t', header=None)
            words_dict = dict(zip(words[0].tolist(), words[1].tolist()))
            self.dictionary[seq_feature] = words_dict
            print('max word id of %s: %d' % (seq_feature, max(self.dictionary[seq_feature].values())))

    def _select_features(self):
        exclude_features = [self.label] + self.seqs_features
        if not self.include_id:
            exclude_features += ['uid', 'iid']
        self.features = [c for c in self.column_max.keys() if c not in exclude_features]
        print("# of features:", len(self.features))

        self.feature_dims = 0
        for f in self.features:
            self.feature_dims += int(self.column_max[f] + 1)
        print("# of feature dims:", self.feature_dims)

        self.user_num, self.item_num = -1, -1
        self.user_features, self.item_features = [], []
        if 'uid' in self.column_max:
            self.user_num = int(self.column_max['uid'] + 1)
            print("# of users:", self.user_num)
            if self.include_id:
                self.user_features = [f for f in self.column_max.keys() if f.startswith('u')]
            else:
                self.user_features = [f for f in self.column_max.keys() if f.startswith('u_')]
            print("# of user features:", len(self.user_features))
        if 'iid' in self.column_max:
            self.item_num = int(self.column_max['iid'] + 1)
            print("# of items:", self.item_num)
            if self.include_id:
                self.item_features = [f for f in self.column_max.keys() if f.startswith('i')]
            else:
                self.item_features = [f for f in self.column_max.keys() if f.startswith('i_')]
            print("# of item features:", len(self.item_features))

    def format_data(self, df):
        df = df.copy()
        if self.label in df.columns:
            data = {'Y': np.array(df[self.label], dtype=np.float32)}
            df.drop([self.label], axis=1, inplace=True)
        else:
            data = {'Y': np.zeros(len(df), dtype=np.float32)}
        ui_id = []
        if self.user_num > 0:
            ui_id.append('uid')
        if self.item_num > 0:
            ui_id.append('iid')
        ui_id = df[ui_id]

        base = 0
        for feature in self.features:
            df[feature] = df[feature].apply(lambda x: x + base)
            base += int(self.column_max[feature] + 1)

        if self.append_id:
            x = pd.concat([ui_id, df[self.features]], axis=1)
            data['X'] = x.values
        else:
            data['X'] = df[self.features].values

        for seq_feature in self.seqs_features:
            seqs = df[seq_feature].str.split(',')
            data[seq_feature + '_length'] = np.array(seqs.apply(lambda x: len(x)), dtype=np.int32)
            max_length = self.column_max[seq_feature]
            if not self.seqs_expand:
                seqs = seqs.apply(lambda x: np.array([int(n) for n in x]))
            else:
                seqs = seqs.apply(lambda x: [int(n) for n in x] + [0] * (max_length - len(x)))
            data[seq_feature] = np.array(seqs.tolist())
            # print(data[seq_feature])
            # print(data[seq_feature + '_length'])
        # data['X'] = np.array(data['X'], dtype=np.float32)
        # print(data['X'])
        # print len(data['Y'])
        return data


def main():
    LoadData('../dataset/', 'test', sep='\t', label='rating', include_id=True, append_id=True,
             seqs_features=['seq'], seqs_sep=',', seqs_expand=True)
    # data = LoadData('../dataset/', 'ml-100k-ci', label=args.label, sep=args.sep, append_id=True, include_id=False)
    return


if __name__ == '__main__':
    main()
