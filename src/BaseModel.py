import tensorflow as tf
import numpy as np
from time import time
import copy
import argparse
from LoadData import LoadData
from tensorflow.contrib.layers.python.layers import batch_norm
import gc
import os
from tqdm import tqdm


class BaseArgs(object):
    def __init__(self, model_name='BaseModel'):
        self.model_name = model_name
        self.parser = argparse.ArgumentParser(description="Run %s." % model_name)
        self.args = self.parse_args()

    def parse_args(self):
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='Set CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--path', nargs='?', default='../dataset/',
                                 help='Input data path.')
        self.parser.add_argument('--model', nargs='?',
                                 default='../model/%s/%s.ckpt' % (self.model_name, self.model_name),
                                 help='Model save path.')
        self.parser.add_argument('--dataset', nargs='?', default='ml-100k-r',
                                 help='Choose a dataset.')
        self.parser.add_argument('--label', type=str, default='rating',
                                 help='name of dataset label column.')
        self.parser.add_argument('--sep', type=str, default=',',
                                 help='sep of csv file.')
        self.parser.add_argument('--verbose', type=int, default=1,
                                 help='Show the results per X epochs (0, 1 ... any positive integer)')
        self.parser.add_argument('--random_seed', type=int, default=2018,
                                 help='Random seed of numpy and tensorflow.')
        self.parser.add_argument('--optimizer', type=str, default='Adagrad',
                                 help='optimizer: GD, Adam, Adagrad')

        self.parser.add_argument('--load', type=int, default=0,
                                 help='Whether load model and continue to train')
        self.parser.add_argument('--epoch', type=int, default=1000,
                                 help='Number of epochs.')
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='Learning rate.')
        self.parser.add_argument('--batch_size', type=int, default=128,
                                 help='Batch size.')
        self.parser.add_argument('--dropout_keep', type=float, default=0.8,
                                 help='Keep probability (i.e., 1-dropout_ratio) for each deep layer')
        self.parser.add_argument('--l2', type=float, default=0.0,
                                 help='Weight of l2_regularize in loss.')
        return self.parser.parse_args()


class BaseModel(object):
    def __init__(self, feature_num,
                 optimizer, epoch, learning_rate, batch_size, dropout_keep, l2_regularize,
                 verbose, random_seed, model_path):

        self.feature_num = feature_num

        self.optimizer_name = optimizer
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.keep_prob = dropout_keep
        self.no_dropout = 1.0
        self.l2_regularize = l2_regularize

        self.model_path = model_path
        self.random_seed = random_seed
        self.d_type = tf.float32
        self.verbose = verbose

        self._init_graph()
        self.train_loss, self.valid_loss, self.test_loss = [], [], []
        self._init_optimizer()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.debug = None
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * [u_id, i_id, u_fs, i_fs]
            self.train_labels = tf.placeholder(dtype=self.d_type, shape=[None])  # None
            self.dropout_keep = tf.placeholder(dtype=self.d_type)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._init_weights()
            self.prediction = tf.add(tf.matmul(tf.cast(self.train_features, self.d_type), self.weights['w']),
                                     self.weights['b'])
            self.prediction = tf.reduce_sum(self.prediction, axis=1)
            # self.debug = self.prediction
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.train_labels, self.prediction))))
            self.loss = self.rmse + self.l2_regularize * tf.reduce_sum(tf.square(self.weights['w']))
            self.var_list = list(tf.trainable_variables())

    def _init_optimizer(self):
        with self.graph.as_default():
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, var_list=self.var_list)
            if self.optimizer_name == 'Adagrad':
                # print("optimizer:", "Adagrad")
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate, initial_accumulator_value=1e-8) \
                    .minimize(self.loss, var_list=self.var_list)
            elif self.optimizer_name == 'Adam':
                # print("optimizer:", "Adam")
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8) \
                    .minimize(self.loss, var_list=self.var_list)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
            #     self.loss)
            # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95,
            #                                             epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            self.sess = tf.Session(config=config)
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_weights(self):
        all_weights = dict()
        all_weights['w'] = tf.Variable(tf.random_normal([self.feature_num, 1], 0.0, 0.01, dtype=self.d_type), name='w')
        all_weights['b'] = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01, dtype=self.d_type), name='b')
        return all_weights

    @staticmethod
    def _batch_norm_layer(x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    @staticmethod
    def shuffle_in_unison_scary(data):
        rng_state = np.random.get_state()
        for d in data:
            np.random.set_state(rng_state)
            np.random.shuffle(d)

    @staticmethod
    def eva_termination(valid):
        if len(valid) > 20 and valid[-1] >= valid[-2] >= valid[-3] >= valid[-4] >= valid[-5]:
            return True
        return False

    @staticmethod
    def evaluate_method(p, l):
        return np.sqrt(np.mean(np.square(p - l)))

    @staticmethod
    def format_metric(metric):
        if type(metric) is not tuple or type(metric) is not list:
            metric = [metric]
        format_str = []
        if type(metric) is tuple or type(metric) is list:
            for m in metric:
                if type(m) in [float, np.float, np.float32, np.float64]:
                    format_str.append('%.4f' % m)
                elif type(m) in [int, np.int, np.int32, np.int64]:
                    format_str.append('%d' % m)
        return ','.join(format_str)

    def load_model(self):
        self.saver.restore(self.sess, self.model_path)

    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def get_feed_dict(self, data, batch_i, batch_size, train):
        batch_start = batch_i * batch_size
        batch_end = min(len(data['Y']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        feed_dict = {self.train_features: data['X'][batch_start: batch_start + real_batch_size],
                     self.train_labels: data['Y'][batch_start:batch_start + real_batch_size],
                     self.dropout_keep: self.keep_prob if train else self.no_dropout, self.train_phase: train}
        return feed_dict

    def predict(self, data):  # evaluate the results for an input set
        if type(data) is list:
            ys = [d['Y'] for d in data]
            xs = [d['X'] for d in data]
            data = {'X': np.concatenate(xs), 'Y': np.concatenate(ys)}
        num_example = len(data['Y'])
        eval_batch_size = self.batch_size * 128
        total_batch = int((num_example + eval_batch_size - 1) / eval_batch_size)
        if num_example == 0:
            return -1.0
        gc.collect()
        predictions = []
        for batch in tqdm(range(total_batch), leave=False):
            feed_dict = self.get_feed_dict(data, batch, eval_batch_size, False)
            prediction = self.sess.run(self.prediction, feed_dict=feed_dict)
            predictions.append(prediction)
        predictions = np.concatenate(predictions)
        gc.collect()
        return predictions

    def evaluate(self, data):  # evaluate the results for an input set
        if type(data) is list:
            ys = [d['Y'] for d in data]
            xs = [d['X'] for d in data]
            data = {'X': np.concatenate(xs), 'Y': np.concatenate(ys)}
        predictions = self.predict(data=data)
        labels = data['Y']
        return self.evaluate_method(predictions, labels)

    def fit(self, data):  # fit the results for an input set
        num_example = len(data['Y'])
        total_batch = int((num_example + self.batch_size - 1) / self.batch_size)
        gc.collect()
        predictions = []
        for batch in tqdm(range(total_batch), leave=False):    
            feed_dict = self.get_feed_dict(data, batch, self.batch_size, True)
            prediction, opt = self.sess.run((self.prediction, self.optimizer), feed_dict=feed_dict)
            predictions.append(prediction)
        predictions = np.concatenate(predictions)
        labels = data['Y']
        gc.collect()
        return self.evaluate_method(predictions, labels)

    def train(self, train_data, validation_data, test_data, load=False):  # fit a dataset
        train_data, validation_data, test_data = \
            copy.deepcopy(train_data), copy.deepcopy(validation_data), copy.deepcopy(test_data)
        if not load:
            self.save_model()
        else:
            self.load_model()
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(train_data) if train_data is not None else -1.0
            init_valid = self.evaluate(validation_data) if validation_data is not None else -1.0
            init_test = self.evaluate(test_data) if test_data is not None else -1.0
            print("Init: \t train=%s, validation=%s, test=%s [%.1f s]" % (
                self.format_metric(init_train), self.format_metric(init_valid), self.format_metric(init_test),
                time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(list(train_data.values()))
            self.fit(train_data)
            t2 = time()

            # output validation
            train_result = self.evaluate(train_data) if train_data is not None else -1.0
            valid_result = self.evaluate(validation_data) if validation_data is not None else -1.0
            test_result = self.evaluate(test_data) if test_data is not None else -1.0

            self.train_loss.append(train_result)
            self.valid_loss.append(valid_result)
            self.test_loss.append(test_result)
            best_valid_score = min(self.valid_loss)
            if best_valid_score == self.valid_loss[-1]:
                self.save_model()
            # self.run_debug(test_data)
            # self.saver.save(self.sess, self.model_path)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\t train=%s, validation=%s, test=%s [%.1f s]"
                      % (epoch + 1, t2 - t1, self.format_metric(train_result),
                         self.format_metric(valid_result), self.format_metric(test_result),
                         time() - t2))
            if self.eva_termination(self.valid_loss):
                print("Early stop at %d based on validation result." % (epoch + 1))
                break
        self.load_model()

    def run_debug(self, data):
        if self.debug is None:
            return None
        if type(data) is list:
            ys = [d['Y'] for d in data]
            xs = [d['X'] for d in data]
            data = {'X': np.concatenate(xs), 'Y': np.concatenate(ys)}
        num_example = len(data['Y'])
        total_batch = int((num_example + self.batch_size - 1) / self.batch_size)
        if num_example == 0:
            return -1.0

        debugs = []
        for batch in range(total_batch):
            gc.collect()
            feed_dict = self.get_feed_dict(data, batch, self.batch_size, False)
            tmp = self.sess.run(self.debug, feed_dict=feed_dict)
            debugs.append(tmp)
            if batch == 0:
                print(tmp)
                print(tmp.shape)
        debugs = np.concatenate(debugs)
        print(debugs)
        print(debugs.shape)
        return debugs

    def print_result(self, t1):
        # Find the best validation result across iterations
        best_valid_score = min(self.valid_loss)
        best_epoch = self.valid_loss.index(best_valid_score)
        print("Best Iter(validation)= %d\t train = %s, valid = %s, test = %s [%.1f s]"
              % (best_epoch + 1, self.format_metric(self.train_loss[best_epoch]),
                 self.format_metric(self.valid_loss[best_epoch]), self.format_metric(self.test_loss[best_epoch]),
                 time() - t1))
        best_test_score = min(self.test_loss)
        best_epoch = self.test_loss.index(best_test_score)
        print("Best Iter(test)= %d\t train = %s, valid = %s, test = %s [%.1f s]"
              % (best_epoch + 1, self.format_metric(self.train_loss[best_epoch]),
                 self.format_metric(self.valid_loss[best_epoch]), self.format_metric(self.test_loss[best_epoch]),
                 time() - t1))


def main():
    # Data loading
    arg_parser = BaseArgs()
    args = arg_parser.args

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    data = LoadData(args.path, args.dataset, label=args.label, append_id=False, include_id=True)
    if args.verbose > 0:
        print(args)

    # Training
    t1 = time()
    model = BaseModel(feature_num=len(data.features), optimizer=args.optimizer,
                      learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch,
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
