import tensorflow as tf
import numpy as np
from time import time
import argparse
import copy
from LoadData import LoadData
from tensorflow.contrib.layers.python.layers import batch_norm
from CSRecModel import CSRecModel, CSRecArgs
import os


class CSACCMArgs(CSRecArgs):
    def __init__(self, model_name='CSACCM'):
        CSRecArgs.__init__(self, model_name)

    def parse_args(self):
        self.parser.add_argument('--f_vector_size', type=int, default=64,
                                 help='Size of feature vectors.')
        self.parser.add_argument('--cb_hidden_layers', nargs='?', default='[]',
                                 help="Number of CB part's hidden layer.")
        self.parser.add_argument('--attention_size', type=int, default=16,
                                 help='Size of attention layer.')
        CSRecArgs.parse_args(self)
        return self.parser.parse_args()


class CSACCM(CSRecModel):
    def __init__(self, feature_num, user_feature_num, item_feature_num, feature_dims, f_vector_size,
                 user_num, item_num, ui_vector_size, warm_ratio,
                 cb_hidden_layers, attention_size,
                 optimizer, epoch, learning_rate, batch_size, dropout_keep, l2_regularize,
                 verbose, random_seed, model_path):
        self.user_feature_num = user_feature_num
        self.item_feature_num = item_feature_num
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.cb_hidden_layers = cb_hidden_layers
        self.attention_size = attention_size
        CSRecModel.__init__(self, feature_num=feature_num, user_num=user_num, item_num=item_num,
                            ui_vector_size=ui_vector_size, warm_ratio=warm_ratio,
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

            self.u_bias = tf.nn.embedding_lookup(self.weights['user_bias'], u_ids) * (1 - self.drop_u_pos)
            self.i_bias = tf.nn.embedding_lookup(self.weights['item_bias'], i_ids) * (1 - self.drop_i_pos)
            # self.debug = self.u_bias

            # cf part
            cf_u_vectors = tf.nn.embedding_lookup(self.weights['uid_embeddings'], u_ids)
            cf_i_vectors = tf.nn.embedding_lookup(self.weights['iid_embeddings'], i_ids)
            self.debug = cf_i_vectors

            drop_u_pos = tf.reshape(self.drop_u_pos, shape=[-1, 1])
            drop_i_pos = tf.reshape(self.drop_i_pos, shape=[-1, 1])
            cf_u_vectors = self.random_u_vectors * drop_u_pos + cf_u_vectors * (1 - drop_u_pos)
            cf_i_vectors = self.random_i_vectors * drop_i_pos + cf_i_vectors * (1 - drop_i_pos)

            self.cf_prediction = tf.reduce_sum(tf.multiply(cf_u_vectors, cf_i_vectors), axis=1)
            self.cf_vector = tf.concat(values=[cf_u_vectors, cf_i_vectors], axis=1)

            # cb part
            u_fs = self.train_features[:, 2:2 + self.user_feature_num]
            i_fs = self.train_features[:, 2 + self.user_feature_num:]
            uf_vectors = tf.nn.embedding_lookup(self.weights['feature_embeddings'], u_fs)
            if_vectors = tf.nn.embedding_lookup(self.weights['feature_embeddings'], i_fs)
            uf_layer = tf.reshape(uf_vectors, (-1, self.f_vector_size * self.user_feature_num))
            if_layer = tf.reshape(if_vectors, (-1, self.f_vector_size * self.item_feature_num))
            for i in range(0, len(self.cb_hidden_layers) + 1):
                uf_layer = tf.add(tf.matmul(uf_layer, self.weights['cb_user_layer_%d' % i]),
                                  self.weights['cb_user_bias_%d' % i])
                uf_layer = self._batch_norm_layer(uf_layer, train_phase=self.train_phase, scope_bn='u_bn_%d' % i)
                uf_layer = tf.nn.relu(uf_layer)
                uf_layer = tf.nn.dropout(uf_layer, self.dropout_keep)

                if_layer = tf.add(tf.matmul(if_layer, self.weights['cb_item_layer_%d' % i]),
                                  self.weights['cb_item_bias_%d' % i])
                if_layer = self._batch_norm_layer(if_layer, train_phase=self.train_phase, scope_bn='i_bn_%d' % i)
                if_layer = tf.nn.relu(if_layer)
                if_layer = tf.nn.dropout(if_layer, self.dropout_keep)
            cb_u_vectors, cb_i_vectors = uf_layer, if_layer
            self.cb_prediction = tf.reduce_sum(tf.multiply(cb_u_vectors, cb_i_vectors), axis=1)
            self.cb_vector = tf.concat(values=[cb_u_vectors, cb_i_vectors], axis=1)

            # attention
            ah_cf_u = tf.add(tf.matmul(cf_u_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            # ah_cf_u = tf.tanh(ah_cf_u)
            # ah_cf_u = tf.nn.relu(ah_cf_u)
            a_cf_u = tf.reduce_sum(tf.multiply(ah_cf_u, self.weights['attention_pre']), axis=1)
            # a_cf_u = tf.minimum(tf.maximum(a_cf_u, -10), 10)
            a_cf_u = tf.exp(a_cf_u)
            ah_cb_u = tf.add(tf.matmul(cb_u_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            # ah_cb_u = tf.tanh(ah_cb_u)
            # ah_cb_u = tf.nn.relu(ah_cb_u)
            a_cb_u = tf.reduce_sum(tf.multiply(ah_cb_u, self.weights['attention_pre']), axis=1)
            a_cb_u = tf.exp(a_cb_u)
            # a_cb_u = tf.minimum(tf.maximum(a_cb_u, -10), 10)
            a_sum = a_cf_u + a_cb_u

            self.a_cf_u, self.a_cb_u = a_cf_u / a_sum, a_cb_u / a_sum

            ah_cf_i = tf.add(tf.matmul(cf_i_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            # ah_cf_i = tf.tanh(ah_cf_i)
            # ah_cf_i = tf.nn.relu(ah_cf_i)
            a_cf_i = tf.reduce_sum(tf.multiply(ah_cf_i, self.weights['attention_pre']), axis=1)
            # a_cf_i = tf.minimum(tf.maximum(a_cf_i, -10), 10)
            a_cf_i = tf.exp(a_cf_i)
            ah_cb_i = tf.add(tf.matmul(cb_i_vectors, self.weights['attention_weights']),
                             self.weights['attention_bias'])
            # ah_cb_i = tf.tanh(ah_cb_i)
            # ah_cb_i = tf.nn.relu(ah_cb_i)
            a_cb_i = tf.reduce_sum(tf.multiply(ah_cb_i, self.weights['attention_pre']), axis=1)
            a_cb_i = tf.exp(a_cb_i)
            # a_cb_i = tf.minimum(tf.maximum(a_cb_i, -10), 10)
            a_sum = a_cf_i + a_cb_i

            self.a_cf_i, self.a_cb_i = a_cf_i / a_sum, a_cb_i / a_sum

            # prediction
            self.bias = self.u_bias + self.i_bias + self.weights['global_bias']

            # self.debug = cf_u_vectors
            self.u_vector = tf.reshape(self.a_cf_u, shape=[-1, 1]) * cf_u_vectors + \
                            tf.reshape(self.a_cb_u, shape=[-1, 1]) * cb_u_vectors
            self.i_vector = tf.reshape(self.a_cf_i, shape=[-1, 1]) * cf_i_vectors + \
                            tf.reshape(self.a_cb_i, shape=[-1, 1]) * cb_i_vectors
            self.prediction = self.bias + tf.reduce_sum(tf.multiply(self.u_vector, self.i_vector), axis=1)
            # self.prediction = tf.sigmoid(self.prediction)

            not_cold_pos = (1 - drop_u_pos) * (1 - drop_i_pos)
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.train_labels, self.prediction))))

            self.rmse_cb = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.train_labels, self.bias + self.cb_prediction))))
            self.rmse_cf = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.train_labels, self.bias + self.cf_prediction) * not_cold_pos)))
            # self.loss = self.rmse + self.difference_weight * self.difference
            self.loss = self.rmse + self.rmse_cb + self.rmse_cf + self.l2_regularize * (
                    tf.reduce_sum(tf.square(self.weights['feature_embeddings'])) +
                    tf.reduce_sum(tf.square(self.weights['uid_embeddings'])) +
                    tf.reduce_sum(tf.square(self.weights['iid_embeddings'])))

            self.var_list = list(tf.trainable_variables())

    def _init_weights(self):
        all_weights = dict()
        all_weights['user_bias'] = tf.Variable(tf.constant(0.0, shape=[self.user_num], dtype=self.d_type))
        all_weights['item_bias'] = tf.Variable(tf.constant(0.0, shape=[self.item_num], dtype=self.d_type))
        all_weights['global_bias'] = tf.Variable(0.1, dtype=self.d_type)

        all_weights['uid_embeddings'] = tf.Variable(
            tf.random_normal([self.user_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='uid_embeddings')  # user_num * ui_vector_size
        all_weights['iid_embeddings'] = tf.Variable(
            tf.random_normal([self.item_num, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='iid_embeddings')  # item_num * ui_vector_size

        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_dims, self.f_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='feature_embeddings')  # feature_dims * f_vector_size

        user_pre_size = self.user_feature_num * self.f_vector_size
        for i, layer_size in enumerate(self.cb_hidden_layers):
            all_weights['cb_user_layer_%d' % i] = tf.Variable(
                tf.random_normal([user_pre_size, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='cb_user_layer_%d' % i)
            all_weights['cb_user_bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='cb_user_bias_%d' % i)
            user_pre_size = self.cb_hidden_layers[i]
        all_weights['cb_user_layer_%d' % len(self.cb_hidden_layers)] = tf.Variable(
            tf.random_normal([user_pre_size, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='cb_user_layer_%d' % len(self.cb_hidden_layers))
        all_weights['cb_user_bias_%d' % len(self.cb_hidden_layers)] = tf.Variable(
            tf.random_normal([1, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='cb_user_bias_%d' % len(self.cb_hidden_layers))

        item_pre_size = self.item_feature_num * self.f_vector_size
        for i, layer_size in enumerate(self.cb_hidden_layers):
            all_weights['cb_item_layer_%d' % i] = tf.Variable(
                tf.random_normal([item_pre_size, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='cb_item_layer_%d' % i)
            all_weights['cb_item_bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.cb_hidden_layers[i]], 0.0, 0.01, dtype=self.d_type),
                name='cb_item_bias_%d' % i)
            item_pre_size = self.cb_hidden_layers[i]
        all_weights['cb_item_layer_%d' % len(self.cb_hidden_layers)] = tf.Variable(
            tf.random_normal([item_pre_size, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='cb_item_layer_%d' % len(self.cb_hidden_layers))
        all_weights['cb_item_bias_%d' % len(self.cb_hidden_layers)] = tf.Variable(
            tf.random_normal([1, self.ui_vector_size], 0.0, 0.01, dtype=self.d_type),
            name='cb_item_bias_%d' % len(self.cb_hidden_layers))

        all_weights['attention_weights'] = tf.Variable(
            tf.random_normal([self.ui_vector_size, self.attention_size], 0.0, 0.01, dtype=self.d_type),
            name='attention_weights')
        all_weights['attention_bias'] = tf.Variable(
            tf.random_normal([1, self.attention_size], 0.0, 0.01, dtype=self.d_type),
            name='attention_bias')
        all_weights['attention_pre'] = tf.Variable(
            tf.random_normal([self.attention_size], 0.0, 0.01, dtype=self.d_type),
            name='attention_pre')
        return all_weights


def main():
    # Data loading
    arg_parser = CSACCMArgs()
    args = arg_parser.args
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    data = LoadData(args.path, args.dataset, label=args.label, sep=args.sep, append_id=True, include_id=False)
    if args.verbose > 0:
        print(args)

    # Training
    t1 = time()
    model = CSACCM(feature_num=len(data.features), feature_dims=data.feature_dims, f_vector_size=args.f_vector_size,
                   user_feature_num=len(data.user_features), item_feature_num=len(data.item_features),
                   user_num=data.user_num, item_num=data.item_num, ui_vector_size=args.ui_vector_size,
                   warm_ratio=args.warm_ratio,
                   cb_hidden_layers=eval(args.cb_hidden_layers), attention_size=args.attention_size,
                   optimizer=args.optimizer, learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch,
                   dropout_keep=args.dropout_keep, l2_regularize=args.l2,
                   verbose=args.verbose, random_seed=args.random_seed, model_path=args.model)
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
