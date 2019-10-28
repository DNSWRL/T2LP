from models import *
from helper import *
from random import *
from pprint import pprint
import pandas as pd
import scipy.sparse as sp
import uuid, sys, os, time, argparse
import pickle, pdb, operator, random, sys
import tensorflow as tf
from collections import defaultdict as ddict
from sklearn.metrics import precision_recall_fscore_support

YEARMIN = -50
YEARMAX = 3000


class T2LP(Model):
    def read_valid(self, filename):
        valid_triples = []
        with open(filename, 'r') as filein:
            temp = []
            for line in filein:
                temp = [int(x.strip()) for x in line.split()[0:3]]
                temp.append([line.split()[3], line.split()[4]])
                valid_triples.append(temp)
        return valid_triples

    def getBatches(self, data, shuffle=True):
        if shuffle: random.shuffle(data)
        num_batches = len(data) // self.p.batch_size

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            yield data[start_idx: start_idx + self.p.batch_size]

    def create_year2id(self, triple_time):
        year2id = dict()
        freq = ddict(int)
        count = 0
        year_list = []

        for k, v in triple_time.items():
            try:
                start = v[0].split('-')[0]  # 起始年
                end = v[1].split('-')[0]  # 结束年
            except:
                pdb.set_trace()

            if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))  # 当时间完整才追加
            if end.find('#') == -1 and len(end) == 4: year_list.append(int(end))

        year_list.sort()
        for year in year_list:
            freq[year] = freq[year] + 1  # 记录每个年份出现的次数

        year_class = []
        count = 0
        for key in sorted(freq.keys()):
            count += freq[key]
            if count > 300:
                year_class.append(key)
                count = 0
        prev_year = 0
        i = 0
        for i, yr in enumerate(year_class):
            year2id[(prev_year, yr)] = i
            prev_year = yr + 1
        year2id[(prev_year, max(year_list))] = i + 1
        self.year_list = year_list

        return year2id

    def get_span_ids(self, start, end):
        start = int(start)
        end = int(end)
        if start > end:
            end = YEARMAX

        if start == YEARMIN:
            start_lbl = 0
        else:
            for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                if start >= key[0] and start <= key[1]:
                    start_lbl = lbl

        if end == YEARMAX:
            end_lbl = len(self.year2id.keys()) - 1
        else:
            for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                if end >= key[0] and end <= key[1]:
                    end_lbl = lbl
        return start_lbl, end_lbl

    def create_id_labels(self, triple_time, dtype):
        YEARMAX = 3000  # 结束年份缺省值
        YEARMIN = -50  # 起始年份缺省值

        inp_idx, start_idx, end_idx = [], [], []

        for k, v in triple_time.items():
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
            if start == '####':
                start = YEARMIN
            elif start.find('#') != -1 or len(start) != 4:
                continue

            if end == '####':
                end = YEARMAX
            elif end.find('#') != -1 or len(end) != 4:
                continue

            start = int(start)
            end = int(end)

            if start > end:
                end = YEARMAX
            inp_idx.append(k)
            if start == YEARMIN:
                start_idx.append(0)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if start >= key[0] and start <= key[1]:
                        start_idx.append(lbl)

            if end == YEARMAX:
                end_idx.append(len(self.year2id.keys()) - 1)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if end >= key[0] and end <= key[1]:
                        end_idx.append(lbl)

        return inp_idx, start_idx, end_idx

    """加载数据"""

    def load_data(self):
        triple_set = []
        with open(self.p.triple2id, 'r') as filein:  # 遍历triple2id数据集，并将三元组存放在triple_set中
            for line in filein:
                # string.strip():移除字符串头尾多余的字符如空格、换位符、换行符等
                tup = (int(line.split()[0].strip()), int(line.split()[1].strip()), int(line.split()[2].strip()))
                triple_set.append(tup)  # 追加到triple_set列表中
        triple_set = set(triple_set)  # 转换成集合set：去除重复的元素

        train_triples = []
        """下列变量均为字典（set）类型"""
        self.start_time, self.end_time, self.num_class = ddict(dict), ddict(dict), ddict(dict)
        triple_time, entity_time = dict(), dict()  # 时间信息应该是不允许重复的
        self.inp_idx, self.start_idx, self.end_idx, self.labels = ddict(list), ddict(list), ddict(list), ddict(list)
        max_ent, max_rel, count = 0, 0, 0

        with open(self.p.dataset, 'r') as filein:  # 遍历train.txt训练集
            for line in filein:
                train_triples.append(
                    [int(x.strip()) for x in line.split()[0:3]])  # 并将<head,rel,tail>三元组存在train_triples中
                triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]  # 将时间信息存储在triple_time中
                count += 1

        with open(self.p.entity2id, 'r', encoding="utf-8") as filein2:  # 遍历entity2id数据集，记录实体总数
            for line in filein2:
                max_ent = max_ent + 1  # 记录entity数量

        self.year2id = self.create_year2id(triple_time)  # 收集时间并整理得到year2id
        self.inp_idx['triple'], self.start_idx['triple'], self.end_idx['triple'] = self.create_id_labels(triple_time,
                                                                                                         'triple')  # 优化时间信息
        # pdb.set_trace()
        print('inp_idx:' + str(len(self.inp_idx['triple'])) + ' start_idx:' + str(
            len(self.start_idx['triple'])) + ' end_idx:' + str(len(self.end_idx['triple'])))
        for i, ele in enumerate(self.inp_idx['entity']):
            if self.start_idx['entity'][i] > self.end_idx['entity'][i]:
                print(self.inp_idx['entity'][i], self.start_idx['entity'][i], self.end_idx['entity'][i])
        self.num_class = len(self.year2id.keys())

        """
		初始化三元组：head-rel-tail
		"""
        keep_idx = set(self.inp_idx['triple'])
        for i in range(len(train_triples) - 1, -1, -1):
            if i not in keep_idx:
                del train_triples[i]

        with open(self.p.relation2id, 'r') as filein3:
            for line in filein3:
                max_rel = max_rel + 1
        index = randint(1, len(train_triples)) - 1

        posh, rela, post = zip(*train_triples)
        head, rel, tail = zip(*train_triples)

        posh = list(posh)
        post = list(post)
        rela = list(rela)

        head = list(head)
        tail = list(tail)
        rel = list(rel)

        for i in range(len(posh)):
            if self.start_idx['triple'][i] < self.end_idx['triple'][i]:
                for j in range(self.start_idx['triple'][i] + 1, self.end_idx['triple'][i] + 1):
                    head.append(posh[i])
                    rel.append(rela[i])
                    tail.append(post[i])
                    self.start_idx['triple'].append(j)

        self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time = [], [], [], [], [], []
        for triple in range(len(head)):
            neg_set = set()
            for k in range(self.p.M):
                possible_head = randint(0, max_ent - 1)
                while (possible_head, rel[triple], tail[triple]) in triple_set or (
                possible_head, rel[triple], tail[triple]) in neg_set:
                    possible_head = randint(0, max_ent - 1)
                self.nh.append(possible_head)
                self.nt.append(tail[triple])
                self.r.append(rel[triple])
                self.ph.append(head[triple])
                self.pt.append(tail[triple])
                self.triple_time.append(self.start_idx['triple'][triple])
                neg_set.add((possible_head, rel[triple], tail[triple]))

        for triple in range(len(tail)):
            neg_set = set()
            for k in range(self.p.M):
                possible_tail = randint(0, max_ent - 1)
                while (head[triple], rel[triple], possible_tail) in triple_set or (
                head[triple], rel[triple], possible_tail) in neg_set:
                    possible_tail = randint(0, max_ent - 1)
                self.nh.append(head[triple])
                self.nt.append(possible_tail)
                self.r.append(rel[triple])
                self.ph.append(head[triple])
                self.pt.append(tail[triple])
                self.triple_time.append(self.start_idx['triple'][triple])
                neg_set.add((head[triple], rel[triple], possible_tail))


        self.max_rel = max_rel
        self.max_ent = max_ent
        self.max_time = len(self.year2id.keys())
        self.data = list(zip(self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time))
        self.data = self.data + self.data[0:self.p.batch_size]

    def calculated_score_for_positive_elements(self, t, epoch, f_valid, eval_mode='valid'):
        loss = np.zeros(self.max_ent)
        start_trip = t[3][0].split('-')[0]
        end_trip = t[3][1].split('-')[0]
        if start_trip == '####':
            start_trip = YEARMIN
        elif start_trip.find('#') != -1 or len(start_trip) != 4:
            return

        if end_trip == '####':
            end_trip = YEARMAX
        elif end_trip.find('#') != -1 or len(end_trip) != 4:
            return

        start_lbl, end_lbl = self.get_span_ids(start_trip, end_trip)
        if eval_mode == 'test':
            f_valid.write(str(t[0]) + '\t' + str(t[1]) + '\t' + str(t[2]) + '\n')
        elif eval_mode == 'valid' and epoch == self.p.test_freq:
            f_valid.write(str(t[0]) + '\t' + str(t[1]) + '\t' + str(t[2]) + '\n')

        pos_head = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                 self.rel: np.array([t[1]]).reshape(-1, 1),
                                                 self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                 self.start_year: np.array([start_lbl] * self.max_ent),
                                                 self.end_year: np.array([end_lbl] * self.max_ent),
                                                 self.mode: -1,
                                                 self.pred_mode: 1,
                                                 self.query_mode: 1})
        pos_head = np.squeeze(pos_head)

        pos_tail = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                 self.rel: np.array([t[1]]).reshape(-1, 1),
                                                 self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                 self.start_year: np.array([start_lbl] * self.max_ent),
                                                 self.end_year: np.array([end_lbl] * self.max_ent),
                                                 self.mode: -1,
                                                 self.pred_mode: -1,
                                                 self.query_mode: 1})
        pos_tail = np.squeeze(pos_tail)

        pos_rel = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                self.rel: np.array([t[1]]).reshape(-1, 1),
                                                self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                self.start_year: np.array([start_lbl] * self.max_rel),
                                                self.end_year: np.array([end_lbl] * self.max_rel),
                                                self.mode: -1,
                                                self.pred_mode: -1,
                                                self.query_mode: -1})
        pos_rel = np.squeeze(pos_rel)

        return pos_head, pos_tail, pos_rel

    def add_placeholders(self):
        self.start_year = tf.placeholder(tf.int32, shape=[None], name='start_time')
        self.end_year = tf.placeholder(tf.int32, shape=[None], name='end_time')
        self.pos_head = tf.placeholder(tf.int32, [None, 1])
        self.pos_tail = tf.placeholder(tf.int32, [None, 1])
        self.rel = tf.placeholder(tf.int32, [None, 1])
        self.neg_head = tf.placeholder(tf.int32, [None, 1])
        self.neg_tail = tf.placeholder(tf.int32, [None, 1])
        self.mode = tf.placeholder(tf.int32, shape=())
        self.pred_mode = tf.placeholder(tf.int32, shape=())
        self.query_mode = tf.placeholder(tf.int32, shape=())

    def create_feed_dict(self, batch, wLabels=True, dtype='train'):
        ph, pt, r, nh, nt, start_idx = zip(*batch)
        feed_dict = {}
        feed_dict[self.pos_head] = np.array(ph).reshape(-1, 1)
        feed_dict[self.pos_tail] = np.array(pt).reshape(-1, 1)
        feed_dict[self.rel] = np.array(r).reshape(-1, 1)
        feed_dict[self.start_year] = np.array(start_idx)
        # feed_dict[self.end_year]   = np.array(end_idx)
        if dtype == 'train':
            feed_dict[self.neg_head] = np.array(nh).reshape(-1, 1)
            feed_dict[self.neg_tail] = np.array(nt).reshape(-1, 1)
            feed_dict[self.mode] = 1
            feed_dict[self.pred_mode] = 0
            feed_dict[self.query_mode] = 0
        else:
            feed_dict[self.mode] = -1

        return feed_dict

    def time_projection(self, data, t):
        inner_prod = tf.tile(tf.expand_dims(tf.reduce_sum(data * t, axis=1), axis=1), [1, self.p.inp_dim])
        prod = t * inner_prod
        data = data - prod
        return data

    def add_model(self):
        # nn_in = self.input_x
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.max_ent, self.p.inp_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  regularizer=self.regularizer)
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.max_rel, self.p.inp_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  regularizer=self.regularizer)
            self.time_embeddings = tf.get_variable(name="time_embedding", shape=[self.max_time, self.p.inp_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        transE_in_dim = self.p.inp_dim
        transE_in = self.ent_embeddings
        ####################------------------------ time aware GCN MODEL ---------------------------##############

        ## Some transE style model ####

        neutral = tf.constant(0)  ## mode = 1 for train mode = -1 test
        test_type = tf.constant(0)  ##  pred_mode = 1 for head -1 for tail
        query_type = tf.constant(0)  ## query mode  =1 for head tail , -1 for rel

        def f_train():
            pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
            pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
            pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
            return pos_h_e, pos_t_e, pos_r_e

        def f_test():
            def head_tail_query():
                def f_head():
                    e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
                    pos_h_e = transE_in
                    pos_t_e = tf.reshape(tf.tile(e2, [self.max_ent]), (self.max_ent, transE_in_dim))
                    return pos_h_e, pos_t_e

                def f_tail():
                    e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
                    pos_h_e = tf.reshape(tf.tile(e1, [self.max_ent]), (self.max_ent, transE_in_dim))
                    pos_t_e = transE_in
                    return pos_h_e, pos_t_e

                pos_h_e, pos_t_e = tf.cond(self.pred_mode > test_type, f_head, f_tail)
                r = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
                pos_r_e = tf.reshape(tf.tile(r, [self.max_ent]), (self.max_ent, transE_in_dim))
                return pos_h_e, pos_t_e, pos_r_e

            def rel_query():
                e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
                e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
                pos_h_e = tf.reshape(tf.tile(e1, [self.max_rel]), (self.max_rel, transE_in_dim))
                pos_t_e = tf.reshape(tf.tile(e2, [self.max_rel]), (self.max_rel, transE_in_dim))
                pos_r_e = self.rel_embeddings
                return pos_h_e, pos_t_e, pos_r_e

            pos_h_e, pos_t_e, pos_r_e = tf.cond(self.query_mode > query_type, head_tail_query, rel_query)
            return pos_h_e, pos_t_e, pos_r_e

        pos_h_e, pos_t_e, pos_r_e = tf.cond(self.mode > neutral, f_train, f_test)
        neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
        neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

        #### ----- time -----###
        t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))

        pos_h_e_t_1 = self.time_projection(pos_h_e, t_1)
        neg_h_e_t_1 = self.time_projection(neg_h_e, t_1)
        pos_t_e_t_1 = self.time_projection(pos_t_e, t_1)
        neg_t_e_t_1 = self.time_projection(neg_t_e, t_1)
        pos_r_e_t_1 = self.time_projection(pos_r_e, t_1)
        # pos_r_e_t_1 = pos_r_e

        if self.p.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims=True)
            neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims=True)
        # self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims=True)
        # self.predict = pos

        '''
		debug_nn([self.pred_mode,self.mode], feed_dict = self.create_feed_dict(self.data[0:self.p.batch_size],dtype='test'))
		'''
        return pos, neg

    def add_loss(self, pos, neg):
        print('pos:' + str(pos))
        print('neg:' + str(neg))
        print('margin:' + str(self.p.margin))
        with tf.name_scope('Loss_op'):
            loss = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
            if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer,
                                                                                        tf.get_collection(
                                                                                            tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self.p.lr)
            train_op = optimizer.minimize(loss)
        time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings, dim=1))
        return train_op

    """模型初始化，params是参数对应外部的args"""

    def __init__(self, params):
        self.p = params  # 将参数映射为p
        self.p.batch_size = self.p.batch_size

        """设置是否正则化，一般可以防止模型过拟合"""
        if self.p.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

        self.load_data()  # 加载并整理数据(head, relation, tail, start, end)
        self.nbatches = len(self.data) // self.p.batch_size  # batch次数 = 数据集数量 // 每次batch的数量
        self.add_placeholders()  # 定义形参
        self.pos, neg = self.add_model()  # 设置模型，正例/反例
        self.loss = self.add_loss(self.pos, neg)  # 设置损失函数
        self.train_op = self.add_optimizer(self.loss)  # 设置优化器
        self.merged_summ = tf.summary.merge_all()  # 将所有summary保存起来
        self.summ_writer = None
        print('模型建立完成')

    def run_epoch(self, sess, data, epoch):
        drop_rate = self.p.dropout

        losses = []
        # total_correct, total_cnt = 0, 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            l, a = sess.run([self.loss, self.train_op], feed_dict=feed)
            losses.append(l)
        return np.mean(losses)

    def fit(self, sess):
        saver = tf.train.Saver(max_to_keep=None)  # 保存模型，且不设置最大模型文件量
        save_dir = 'checkpoints/' + self.p.name + '/'  # 模型存储路径
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_dir_results = './results/' + self.p.name + '/'  # 结果存储路径
        if not os.path.exists(save_dir_results): os.makedirs(save_dir_results)
        if self.p.restore:
            save_path = os.path.join(save_dir, 'epoch_{}'.format(self.p.restore_epoch))
            saver.restore(sess, save_path)

        if not self.p.onlyTest:
            """
			开始训练
			"""
            print('开始拟合...')
            validation_data = self.read_valid(self.p.valid_data)
            for epoch in range(self.p.max_epochs):  # 分批次处理数据
                l = self.run_epoch(sess, self.data, epoch)  # 运行模型并记录损失
                if epoch % 50 == 0:
                    print('Epoch {}\tLoss {}\t model {}'.format(epoch, l, self.p.name))
                if epoch % self.p.test_freq == 0 and epoch != 0:
                    save_path = os.path.join(save_dir, 'epoch_{}'.format(epoch))  ## -- check pointing -- ##
                    saver.save(sess=sess, save_path=save_path)
                    if epoch == self.p.test_freq:
                        f_valid = open(save_dir_results + '/valid.txt', 'w')

                    fileout_head = open(save_dir_results + '/valid_head_pred_{}.txt'.format(epoch), 'w')
                    fileout_tail = open(save_dir_results + '/valid_tail_pred_{}.txt'.format(epoch), 'w')
                    fileout_rel = open(save_dir_results + '/valid_rel_pred_{}.txt'.format(epoch), 'w')
                    for i, t in enumerate(validation_data):
                        score = self.calculated_score_for_positive_elements(t, epoch, f_valid, 'valid')
                        if score:
                            fileout_head.write(' '.join([str(x) for x in score[0]]) + '\n')
                            fileout_tail.write(' '.join([str(x) for x in score[1]]) + '\n')
                            fileout_rel.write(' '.join([str(x) for x in score[2]]) + '\n')

                        if i % 500 == 0:
                            print('{}. no of valid_triples complete'.format(i))

                    fileout_head.close()
                    fileout_tail.close()
                    fileout_rel.close()
                    if epoch == self.p.test_freq:
                        f_valid.close()
                    print("验证结束")
            """
			训练结束
			"""
        else:
            """
			开始测试
			"""
            print('开始测试')
            test_data = self.read_valid(self.p.test_data)
            f_test = open(save_dir_results + '/test.txt', 'w')
            fileout_head = open(save_dir_results + '/test_head_pred_{}.txt'.format(self.p.restore_epoch), 'w')
            fileout_tail = open(save_dir_results + '/test_tail_pred_{}.txt'.format(self.p.restore_epoch), 'w')
            fileout_rel = open(save_dir_results + '/test_rel_pred_{}.txt'.format(self.p.restore_epoch), 'w')
            for i, t in enumerate(test_data):
                score = self.calculated_score_for_positive_elements(t, self.p.restore_epoch, f_test, 'test')
                fileout_head.write(' '.join([str(x) for x in score[0]]) + '\n')
                fileout_tail.write(' '.join([str(x) for x in score[1]]) + '\n')
                fileout_rel.write(' '.join([str(x) for x in score[2]]) + '\n')

                if i % 500 == 0:
                    print('{}. no of test_triples complete'.format(i))
            fileout_head.close()
            fileout_tail.close()
            fileout_rel.close()
            print("测试结束")
            """
			测试结束
			"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T2LP')

    parser.add_argument('-data_type', dest="data_type", default='yago', choices=['yago', 'wiki_data'],
                        help='dataset to choose')
    parser.add_argument('-version', dest='version', default='large', choices=['large', 'small'],
                        help='data version to choose')
    parser.add_argument('-test_freq', dest="test_freq", default=25, type=int, help='Batch size')
    parser.add_argument('-neg_sample', dest="M", default=5, type=int, help='Batch size')
    parser.add_argument('-gpu', dest="gpu", default='1', help='GPU to use')
    parser.add_argument('-name', dest="name", default='test_' + str(uuid.uuid4()), help='Name of the run')
    parser.add_argument('-drop', dest="dropout", default=1.0, type=float, help='Dropout for full connected layer')
    parser.add_argument('-rdrop', dest="rec_dropout", default=1.0, type=float, help='Recurrent dropout for LSTM')
    parser.add_argument('-lr', dest="lr", default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-lam_1', dest="lambda_1", default=0.5, type=float, help='transE weight')
    parser.add_argument('-lam_2', dest="lambda_2", default=0.25, type=float, help='entitty loss weight')
    parser.add_argument('-margin', dest="margin", default=1, type=float, help='margin')
    parser.add_argument('-batch', dest="batch_size", default=50000, type=int, help='Batch size')
    parser.add_argument('-epoch', dest="max_epochs", default=10, type=int, help='Max epochs')
    parser.add_argument('-l2', dest="l2", default=0.0, type=float, help='L2 regularization')
    parser.add_argument('-seed', dest="seed", default=1234, type=int, help='Seed for randomization')
    parser.add_argument('-inp_dim', dest="inp_dim", default=128, type=int, help='Hidden state dimension of Bi-LSTM')
    parser.add_argument('-L1_flag', dest="L1_flag", action='store_false', help='Hidden state dimension of FC layer')
    parser.add_argument('-onlytransE', dest="onlytransE", action='store_true',
                        help='Evaluate model on only transE loss')
    parser.add_argument('-onlyTest', dest="onlyTest", action='store_true', help='Evaluate model for test data')
    parser.add_argument('-restore', dest="restore", action='store_true',
                        help='Restore from the previous best saved model')
    parser.add_argument('-res_epoch', dest="restore_epoch", default=200, type=int,
                        help='Restore from the previous best saved model')
    args = parser.parse_args()
    args.dataset = args.data_type + '/train.txt'
    args.entity2id = args.data_type  + '/entity2id.txt'
    args.relation2id = args.data_type + '/relation2id.txt'
    args.valid_data = args.data_type + '/valid.txt'
    args.test_data = args.data_type + '/test.txt'
    args.triple2id = args.data_type + '/triple2id.txt'

    tf.set_random_seed(args.seed)  # 给tensorflow设置图形级随机seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_gpu(args.gpu)
    model = T2LP(args)  # 模型初始化
    config = tf.ConfigProto()  # 配置session用
    config.gpu_options.allow_growth = True  # 设置按需求占用GPU，防止产生内存碎片浪费资源
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  # 初始化模型参数
        model.fit(sess)  # 拟合模型
