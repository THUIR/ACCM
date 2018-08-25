import tensorflow as tf
import numpy as np
from time import time
import argparse
import copy
from LoadData import LoadData
from tensorflow.contrib.layers.python.layers import batch_norm
from RecModel import RecModel, RecArgs
import os
from tqdm import tqdm
import gc


class CSRecArgs(RecArgs):
    def __init__(self, model_name='CSRecModel'):
        RecArgs.__init__(self, model_name)

    def parse_args(self):
        self.parser.add_argument('--warm_ratio', type=float, default=0.9,
                                 help='Warm ratio of each batch.')
        RecArgs.parse_args(self)
        return self.parser.parse_args()


class CSRecModel(RecModel):
    def __init__(self, feature_num, user_num, item_num, ui_vector_size, warm_ratio,
                 optimizer, epoch, learning_rate, batch_size, dropout_keep, l2_regularize,
                 verbose, random_seed, model_path):
        self.warm_ratio = warm_ratio
        RecModel.__init__(self, feature_num=feature_num, user_num=user_num, item_num=item_num,
                          ui_vector_size=ui_vector_size,
                          optimizer=optimizer, epoch=epoch, learning_rate=learning_rate, batch_size=batch_size,
                          dropout_keep=dropout_keep, l2_regularize=l2_regularize,
                          verbose=verbose, random_seed=random_seed, model_path=model_path)

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.debug = None
            tf.set_random_seed(self.random_seed)

            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * [u_id, i_id, u_fs, i_fs]
            self.train_labels = tf.placeholder(dtype=self.d_type, shape=[None])  # None
            self.drop_u_pos = tf.placeholder(dtype=self.d_type, shape=[None])
            self.drop_i_pos = tf.placeholder(dtype=self.d_type, shape=[None])
            self.random_u_vectors = tf.placeholder(dtype=self.d_type, shape=[None, None])
            self.random_i_vectors = tf.placeholder(dtype=self.d_type, shape=[None, None])
            self.dropout_keep = tf.placeholder(dtype=self.d_type)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._init_weights()

            u_ids = self.train_features[:, 0]
            i_ids = self.train_features[:, 1]

            # cf part
            cf_u_vectors = tf.nn.embedding_lookup(self.weights['uid_embeddings'], u_ids)
            cf_i_vectors = tf.nn.embedding_lookup(self.weights['iid_embeddings'], i_ids)
            self.debug = cf_i_vectors

            drop_u_pos = tf.reshape(self.drop_u_pos, shape=[-1, 1])
            drop_i_pos = tf.reshape(self.drop_i_pos, shape=[-1, 1])
            cf_u_vectors = self.random_u_vectors * drop_u_pos + cf_u_vectors * (1 - drop_u_pos)
            cf_i_vectors = self.random_i_vectors * drop_i_pos + cf_i_vectors * (1 - drop_i_pos)

            self.cf_prediction = tf.reduce_sum(tf.multiply(cf_u_vectors, cf_i_vectors), axis=1)

            # Prediction
            self.prediction = self.cf_prediction
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.train_labels, self.prediction))))
            self.loss = self.rmse + self.l2_regularize * (
                    tf.reduce_sum(tf.square(self.weights['uid_embeddings'])) +
                    tf.reduce_sum(tf.square(self.weights['iid_embeddings'])))
            self.var_list = list(tf.trainable_variables())

    def _init_weights(self):
        all_weights = dict()
        all_weights['uid_embeddings'] = tf.Variable(
            tf.random_normal([self.user_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='uid_embeddings')  # user_num * ui_vector_size
        all_weights['iid_embeddings'] = tf.Variable(
            tf.random_normal([self.item_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='iid_embeddings')  # item_num * ui_vector_size
        return all_weights

    def get_feed_dict(self, data, batch_i, batch_size, train=False):
        batch_start = batch_i * batch_size
        batch_end = min(len(data['Y']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        if train:
            drop_u_pos = np.ones(real_batch_size)
            drop_u_pos[np.random.choice(real_batch_size, int(real_batch_size * self.warm_ratio), replace=False)] = 0
            drop_i_pos = np.ones(real_batch_size)
            drop_i_pos[np.random.choice(real_batch_size, int(real_batch_size * self.warm_ratio), replace=False)] = 0
            random_u_vectors = np.random.normal(0.0, 0.01, size=(real_batch_size, self.ui_vector_size))
            random_i_vectors = np.random.normal(0.0, 0.01, size=(real_batch_size, self.ui_vector_size))
        else:
            drop_u_pos = np.zeros(real_batch_size)
            drop_i_pos = np.zeros(real_batch_size)
            random_u_vectors = np.zeros(shape=(real_batch_size, self.ui_vector_size))
            random_i_vectors = np.zeros(shape=(real_batch_size, self.ui_vector_size))
        feed_dict = {self.train_features: data['X'][batch_start: batch_start + real_batch_size],
                     self.train_labels: data['Y'][batch_start:batch_start + real_batch_size],
                     self.dropout_keep: self.keep_prob if train else self.no_dropout, self.train_phase: train,
                     self.drop_u_pos: drop_u_pos, self.drop_i_pos: drop_i_pos,
                     self.random_u_vectors: random_u_vectors, self.random_i_vectors: random_i_vectors}
        return feed_dict


def main():
    # Data loading
    arg_parser = CSRecArgs()
    args = arg_parser.args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    data = LoadData(args.path, args.dataset, label=args.label, append_id=True, include_id=False)
    if args.verbose > 0:
        print(args)

    # Training
    t1 = time()
    model = CSRecModel(feature_num=len(data.features), user_num=data.user_num, item_num=data.item_num,
                       warm_ratio=args.warm_ratio,
                       optimizer=args.optimizer, learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch,
                       dropout_keep=args.dropout_keep, l2_regularize=args.l2,
                       verbose=args.verbose, random_seed=args.random_seed,
                       ui_vector_size=args.ui_vector_size, model_path=args.model)
    if args.load == 1:
        model.load_model()
    # model.run_debug(data.test_data)

    # train
    model.train(data.train_data, data.validation_data, data.test_data, args.load == 1)
    model.print_result(t1)

    # test
    model.load_model()
    model.predict(data.test_data)


if __name__ == '__main__':
    main()
