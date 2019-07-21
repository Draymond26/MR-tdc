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

class ConvNet():
    
    def __init__(self,
                 n_channel, 
                 n_classes, 
                 image_height, 
                 image_width, 
                 sentence_length,
                 vocab_size,
                 embedding_dim,
                 LSTM_hidden_size):
        
        # 输入变量
        self.images1 = tf.placeholder(
            dtype=tf.float32, shape=[None, image_height, image_width, n_channel], name='images')
        self.images2 = tf.placeholder(
            dtype=tf.float32, shape=[None, image_height, image_width, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        '''
        self.instructions = tf.placeholder(
            dtype=tf.int64, shape=[None, sentence_length], name='instructions')
        '''
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable( 
            0, dtype=tf.int32, name='global_step')
        
        # CNN网络结构
        self.conv_layer1 = ConvLayer(
            input_shape=(None, image_height, image_width, n_channel), n_size=3, n_filter=64, 
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv1')
        self.pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')
        
        self.conv_layer2 = ConvLayer(
            input_shape=(None, int(image_height/2), int(image_width/2), 64), n_size=3, n_filter=128,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv2')
        self.pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        
        self.conv_layer3 = ConvLayer(
            input_shape=(None, int(image_height/4), int(image_width/4), 128), n_size=3, n_filter=256, 
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4, 
            name='conv3')
        self.pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        
        self.dense_layer1 = DenseLayer(
            input_shape=(None, int(image_height/8) * int(image_width/8) * 256), hidden_dim=1024, 
            activation='relu', dropout=True, keep_prob=self.keep_prob, 
            batch_normal=False, weight_decay=1e-4, name='dense1')
        
        self.dense_layer2 = DenseLayer(
            input_shape=(None, 1024), hidden_dim=512,
            activation='none', dropout=False, keep_prob=None, 
            batch_normal=False, weight_decay=1e-4, name='dense2')
        
        # CNN数据流
        cnn_output1 = self.get_output(self.images1, image_height, image_width)
        cnn_output2 = self.get_output(self.images2, image_height, image_width)
        
        '''
        #LSTM
        embedding = tf.get_variable("embedding", [vocab_size, embedding_dim])	# [19, 128]
        input_embeddings = tf.nn.embedding_lookup(embedding, self.instructions)	# [batch_size, 5, 128]
        instruc_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_hidden_size)	# LSTM_hidden_size = 128
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(instruc_cell, input_embeddings, dtype=tf.float32)	#[batch_size, 5, 128]
        lstm_output = lstm_outputs[:, -1, :]	#[batch_size, 128]
        '''
        
        #predict
        fully_connected1 = DenseLayer(
            input_shape=(None, 512), hidden_dim=256, 
            activation='relu', dropout=False, keep_prob=self.keep_prob, 
            batch_normal=False, weight_decay=1e-4, name='fc1')
        fully_connected2 = DenseLayer(
            input_shape=(None, 256), hidden_dim=n_classes, 
            activation='relu', dropout=False, keep_prob=self.keep_prob, 
            batch_normal=False, weight_decay=1e-4, name='fc2')
        
        multi_output = tf.multiply(cnn_output1, cnn_output2, name='element_wise_multiplication')
        output_fc1 = fully_connected1.get_output(input=multi_output)
        logits = fully_connected2.get_output(input=output_fc1)
        logit = tf.argmax(logits, 1, name='predicted_class')
        
        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 5000), 
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 10000), 
                                     lambda: tf.constant(0.001),
                                     lambda: tf.constant(0.0001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        
        # 观察值
        correct_prediction = tf.equal(self.labels, logit)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    def get_output(self, input, image_height, image_width):
        hidden_conv1 = self.conv_layer1.get_output(input=input)
        hidden_pool1 = self.pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = self.conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = self.pool_layer2.get_output(input=hidden_conv2)
        hidden_conv3 = self.conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = self.pool_layer3.get_output(input=hidden_conv3)
        input_dense1 = tf.reshape(hidden_pool3, [-1, int(image_height/8) * int(image_width/8) * 256])
        output_dense1 = self.dense_layer1.get_output(input=input_dense1)
        cnn_output = self.dense_layer2.get_output(input=output_dense1)
        #logit = tf.argmax(logits, 1, name='predicted_class')
        return cnn_output
        
    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=10)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch+1):
            train_images1 = dataloader.train_images1
            train_images2 = dataloader.train_images2
            #train_instructions = dataloader.train_instructions
            train_labels = dataloader.train_labels
            valid_images1 = dataloader.valid_images1
            valid_images2 = dataloader.valid_images2
            #valid_instructions = dataloader.valid_instructions
            valid_labels = dataloader.valid_labels
            
            # 开始本轮的训练，并计算目标函数值
            train_accuracy, train_loss = 0.0, 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images1 = train_images1[i: i+batch_size]
                batch_images2 = train_images2[i: i+batch_size]
                #batch_instructions = train_instructions[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_accuracy, avg_loss, iteration] = self.sess.run(
                    fetches=[self.optimizer, self.accuracy, self.avg_loss, self.global_step], 
                    feed_dict={self.images1: batch_images1, 
                               self.images2: batch_images2, 
                               #self.instructions: batch_instructions,
                               self.labels: batch_labels, 
                               self.keep_prob: 0.5})
                train_accuracy += avg_accuracy * batch_images1.shape[0]
                train_loss += avg_loss * batch_images1.shape[0]
            train_accuracy = 1.0 * train_accuracy / dataloader.n_train
            train_loss = 1.0 * train_loss / dataloader.n_train
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images1 = valid_images1[i: i+batch_size]
                batch_images2 = valid_images2[i: i+batch_size]
                #batch_instructions = valid_instructions[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images1: batch_images1, 
                               self.images2: batch_images2, 
                               #self.instructions: batch_instructions,
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images1.shape[0]
                valid_loss += avg_loss * batch_images1.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid
            
            print('epoch{%d}, iter[%d], train precision: %.6f, train loss: %.6f, '
                  'valid precision: %.6f, valid loss: %.6f' % (
                epoch, iteration, train_accuracy, train_loss, valid_accuracy, valid_loss))
            sys.stdout.flush()
            
            # 保存模型
            if not os.path.exists(backup_path):
                os.makedirs(backup_path)
            saver_path = self.saver.save(self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
                
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
