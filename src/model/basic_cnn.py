# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer
#import ipdb

class ConvNet():

    def __init__(self,
                 n_channel,
                 n_classes,
                 image_height,
                 image_width):

        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_height, image_width, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')

        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(None, image_height, image_width, n_channel), n_size=3, n_filter=64,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')

        conv_layer2 = ConvLayer(
            input_shape=(None, int(image_height/2), int(image_width/2), 64), n_size=3, n_filter=128,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        '''
        conv_layer3 = ConvLayer(
            input_shape=(None, int(image_height/4), int(image_width/4), 128), n_size=3, n_filter=256,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        '''
        dense_layer1 = DenseLayer(
            input_shape=(None, int(image_height/4) * int(image_width/4) * 128), hidden_dim=1024,
            activation='relu', dropout=True, keep_prob=self.keep_prob,
            batch_normal=False, weight_decay=1e-4, name='dense1')

        dense_layer2 = DenseLayer(
            input_shape=(None, 1024), hidden_dim=n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense2')

        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        '''
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv3)
        '''
        input_dense1 = tf.reshape(hidden_pool2, [-1, int(image_height/4) * int(image_width/4) * 128])
        output_dense1 = dense_layer1.get_output(input=input_dense1)
        logits = dense_layer2.get_output(input=output_dense1)
        self.logit = tf.argmax(logits, 1, name='predicted_class')

        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 20),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100),
                                     lambda: tf.constant(0.001),
                                     lambda: tf.constant(0.0001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr*0.001).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        self.correct_prediction = tf.equal(self.labels, self.logit)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def train(self, dataloader, backup_path, n_epoch, batch_size, model_path, n_load_epoch):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2,
            max_to_keep=10)
        # 模型初始化
        if len(model_path)==0 and n_load_epoch==0:
            self.sess.run(tf.global_variables_initializer())
        else:
            model_path = os.path.join(model_path, 'model_%d.ckpt' % (n_load_epoch))
            self.saver.restore(self.sess, model_path)
        pre_valid_accuracy = 0
        # 模型训练
        for epoch in range(0, n_epoch+1):
            train_images = dataloader.train_images
            train_labels = dataloader.train_labels
            train_instructions = dataloader.train_instructions
            valid_images = dataloader.valid_images
            valid_labels = dataloader.valid_labels
            valid_instructions = dataloader.valid_instructions

            # 开始本轮的训练，并计算目标函数值
            train_accuracy, train_loss = 0.0, 0.0
            correct_per_instructions = [0 for i in range(7)]
            total_per_instructions = [0 for i in range(7)]
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                batch_instructions = train_instructions[i: i+batch_size]
                [_, avg_accuracy, avg_loss, iteration, is_correct] = self.sess.run(
                    fetches=[self.optimizer, self.accuracy, self.avg_loss, self.global_step, self.correct_prediction],
                    feed_dict={self.images: batch_images,
                               self.labels: batch_labels,
                               self.keep_prob: 0.5})
                train_accuracy += avg_accuracy * batch_images.shape[0]
                train_loss += avg_loss * batch_images.shape[0]
                for j in range(len(batch_labels)):
                    total_per_instructions[batch_instructions[j]] += 1
                    if is_correct[j]:
                        correct_per_instructions[batch_instructions[j]] += 1
            train_accuracy = 1.0 * train_accuracy / dataloader.n_train
            train_loss = 1.0 * train_loss / dataloader.n_train
            accu_per_instructions = list(map((lambda a, b: float(a)/b), correct_per_instructions, total_per_instructions))

            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            valid_correct_per_instructions = [0 for i in range(7)]
            valid_total_per_instructions = [0 for i in range(7)]
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                batch_instructions = valid_instructions[i: i+batch_size]
                [avg_accuracy, avg_loss, is_correct] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss, self.correct_prediction],
                    feed_dict={self.images: batch_images,
                               self.labels: batch_labels,
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
                for j in range(len(batch_labels)):
                    valid_total_per_instructions[batch_instructions[j]] += 1
                    if is_correct[j]:
                        valid_correct_per_instructions[batch_instructions[j]] += 1
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid
            valid_accu_per_instructions = list(map((lambda a, b: float(a)/b), valid_correct_per_instructions, valid_total_per_instructions))

            print('epoch{%d}, iter[%d], train precision: %.6f, train loss: %.6f, '
                  'valid precision: %.6f, valid loss: %.6f' % (
                epoch+n_load_epoch, iteration, train_accuracy, train_loss, valid_accuracy, valid_loss))
            for i in range(len(accu_per_instructions)):
                print('instruction[%d], train accuracy: %f, test accuracy: %f' %(i, accu_per_instructions[i], valid_accu_per_instructions[i]))
            sys.stdout.flush()

            # 保存模型
            if not os.path.exists(backup_path):
                os.makedirs(backup_path)
            if valid_accuracy>pre_valid_accuracy:
                saver_path = self.saver.save(self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch+n_load_epoch)))
                pre_valid_accuracy = valid_accuracy

        self.sess.close()

    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        accuracy_list = []
        test_images = dataloader.data_augmentation(dataloader.test_images,
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy],
                feed_dict={self.images:batch_images,
                           self.labels:batch_labels,
                           self.keep_prob:1.0})
            accuracy_list.append(avg_accuracy)
        print('test precision: %.4f' % (numpy.mean(accuracy_list)))
        self.sess.close()

    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.observe],
            feed_dict={self.images: numpy.random.random(size=[128, 24, 24, 3]),
                       self.labels: numpy.random.randint(low=0, high=9, size=[128,]),
                       self.keep_prob: 1.0})
        print(temp)

    def observe_salience(self, batch_size=128, image_h=32, image_w=32, n_channel=3,
                         num_test=10, epoch=1):
        if not os.path.exists('results/epoch%d/' % (epoch)):
            os.makedirs('results/epoch%d/' % (epoch))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
        assert(os.path.exists(model_path+'.index'))
        saver.restore(sess, model_path)
        print('read model from %s' % (model_path))
        # 获取图像并计算梯度
        for batch in range(num_test):
            batch_image, batch_label = cifar10.test.next_batch(batch_size)
            image = numpy.array(batch_image.reshape([image_h, image_w, n_channel]) * 255,
                                dtype='uint8')
            result = sess.run([self.labels_prob, self.labels_max_prob, self.labels_pred,
                               self.gradient],
                              feed_dict={self.images:batch_image, self.labels:batch_label,
                                         self.keep_prob:0.5})
            print(result[0:3], result[3][0].shape)
            gradient = sess.run(self.gradient, feed_dict={
                self.images:batch_image, self.keep_prob:0.5})
            gradient = gradient[0].reshape([image_h, image_w, n_channel])
            gradient = numpy.max(gradient, axis=2)
            gradient = numpy.array((gradient - gradient.min()) * 255
                                    / (gradient.max() - gradient.min()), dtype='uint8')
            print(gradient.shape)
            # 使用pyplot画图
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(gradient, cmap=plt.cm.gray)
            plt.savefig('results/epoch%d/result_%d.png' % (epoch, batch))

    def observe_hidden_distribution(self, batch_size=128, image_h=32, image_w=32, n_channel=3,
                                    num_test=10, epoch=1):
        if not os.path.exists('results/epoch%d/' % (epoch)):
            os.makedirs('results/epoch%d/' % (epoch))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sess = tf.Session()
        # 读取模型
        model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
        if os.path.exists(model_path+'.index'):
            saver.restore(sess, model_path)
            print('read model from %s' % (model_path))
        else:
            sess.run(tf.global_variables_initializer())
        # 获取图像并计算梯度
        for batch in range(num_test):
            batch_image, batch_label = cifar10.test.next_batch(batch_size)
            result = sess.run([self.nobn_conv1, self.bn_conv1, self.nobn_conv2, self.bn_conv2,
                               self.nobn_conv3, self.bn_conv3, self.nobn_fc1, self.nobn_fc1,
                               self.nobn_softmax, self.bn_softmax],
                              feed_dict={self.images:batch_image, self.labels:batch_label,
                                         self.keep_prob:0.5})
            distribution1 = result[0][:,0].flatten()
            distribution2 = result[1][:,0].flatten()
            distribution3 = result[2][:,0].flatten()
            distribution4 = result[3][:,0].flatten()
            distribution5 = result[4][:,0].flatten()
            distribution6 = result[5][:,0].flatten()
            distribution7 = result[6][:,0].flatten()
            distribution8 = result[7][:,0].flatten()
            plt.subplot(241)
            plt.hist(distribution1, bins=50, color='#1E90FF')
            plt.title('convolutional layer 1')
            plt.subplot(242)
            plt.hist(distribution3, bins=50, color='#1C86EE')
            plt.title('convolutional layer 2')
            plt.subplot(243)
            plt.hist(distribution5, bins=50, color='#1874CD')
            plt.title('convolutional layer 3')
            plt.subplot(244)
            plt.hist(distribution7, bins=50, color='#5CACEE')
            plt.title('full connection layer')
            plt.subplot(245)
            plt.hist(distribution2, bins=50, color='#00CED1')
            plt.title('batch normalized')
            plt.subplot(246)
            plt.hist(distribution4, bins=50, color='#48D1CC')
            plt.title('batch normalized')
            plt.subplot(247)
            plt.hist(distribution6, bins=50, color='#40E0D0')
            plt.title('batch normalized')
            plt.subplot(248)
            plt.hist(distribution8, bins=50, color='#00FFFF')
            plt.title('batch normalized')
            plt.show()
