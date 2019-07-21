# -*- encoding: utf8 -*-
import numpy as np
import os,sys

class DataLoader:
    
    def __init__(self, data_file, n_channel):
        self.data_file = data_file
        dataset = np.load(self.data_file)
        if n_channel==1:
            self.train_images = np.reshape(dataset['train_image'], (dataset['train_image'].shape[0], dataset['train_image'].shape[1], dataset['train_image'].shape[2], 1))
            self.valid_images = np.reshape(dataset['valid_image'], (dataset['valid_image'].shape[0], dataset['valid_image'].shape[1], dataset['valid_image'].shape[2], 1))
        elif n_channel==3:
            self.train_images = dataset['train_image']
            self.valid_images = dataset['valid_image']
        self.train_labels = dataset['train_label']
        self.valid_labels = dataset['valid_label']
        self.train_instructions = dataset['train_instruction']
        self.valid_instructions = dataset['valid_instruction']
        
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
