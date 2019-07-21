# -*- encoding: utf8 -*-
# author: ronniecao
import os,sys
import time
from src.data.load_data_img_pair_concat import DataLoader

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
start_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
dir = '/home/liuguangze/Experiments/MR-tdc/backup/'
n_channel = 1
n_classes = 4
batch_size = 8
image_height = 168
image_width = 84
n_epoch = 4000
sentence_length = 4
vocab_size = 19
embedding_dim = 512
LSTM_hidden_size = 512
model_path = ''
n_load_epoch = 0
#model_path = '/home/liuguangze/Experiments/MR-tdc/backup/2019-05-12-10:17:42'
#n_load_epoch = 3828

data_file = '/home/liuguangze/Experiments/MR-tdc/generate-shuffled-dataset/dataset.npz'
dataloader = DataLoader(data_file, n_channel)

def basic_cnn(n_channel, n_classes, batch_size, image_height, image_width, sentence_length, vocab_size, embedding_dim, LSTM_hidden_size, model_path, n_load_epoch):
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=n_channel, n_classes=n_classes, image_height=image_height, image_width=image_width)
    # convnet.debug()
    convnet.train(dataloader=dataloader, backup_path=os.path.join(dir, start_time), batch_size=batch_size, n_epoch=n_epoch, model_path=model_path, n_load_epoch=n_load_epoch)
    # convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v2/', epoch=5000, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)


basic_cnn(n_channel, n_classes, batch_size, image_height, image_width, sentence_length, vocab_size, embedding_dim, LSTM_hidden_size, model_path, n_load_epoch)
