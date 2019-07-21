# -*- encoding: utf8 -*-
import numpy as np
import os,sys

class DataLoader:
    
    def __init__(self, data_file, n_channel):
        self.data_file = data_file
        dataset = np.load(self.data_file)
        if n_channel==1:
            self.train_images1 = np.reshape(dataset['train_image1'], (dataset['train_image1'].shape[0], dataset['train_image1'].shape[1], dataset['train_image1'].shape[2], 1))
            self.valid_images1 = np.reshape(dataset['valid_image1'], (dataset['valid_image1'].shape[0], dataset['valid_image1'].shape[1], dataset['valid_image1'].shape[2], 1))
            self.train_images2 = np.reshape(dataset['train_image2'], (dataset['train_image2'].shape[0], dataset['train_image2'].shape[1], dataset['train_image2'].shape[2], 1))
            self.valid_images2 = np.reshape(dataset['valid_image2'], (dataset['valid_image2'].shape[0], dataset['valid_image2'].shape[1], dataset['valid_image2'].shape[2], 1))
        elif n_channel==3:
            self.train_images1 = dataset['train_image1']
            self.valid_images1 = dataset['valid_image1']
            self.train_images2 = dataset['train_image2']
            self.valid_images2 = dataset['valid_image2']
        self.train_labels = dataset['train_label']
        self.valid_labels = dataset['valid_label']
        
        self.n_train = self.train_images1.shape[0]
        self.n_valid = self.valid_images1.shape[0]