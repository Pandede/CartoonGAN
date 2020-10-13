# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:00:22 2019

@author: Chan Chak Tong
"""
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import random
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, UpSampling2D, Dense, Flatten, Input, BatchNormalization, Reshape, LeakyReLU, Dropout

class DataLoader:
    def __init__(self, folder_path, img_size):
        self.folder_path = folder_path
        self.img_size = img_size
        
        self.path_list = glob(folder_path)		# 讀取資料夾全部圖片路徑
    
    def __imread(self, img_path):
        '''讀取圖片'''
        return np.array(Image.open(img_path).convert('RGB').resize(self.img_size[:-1], Image.LANCZOS))
    
    def sampling_data(self, batch_size, shuffle=True):
        img_path_list = self.path_list
        
        if shuffle:
            random.shuffle(img_path_list)
        
        for batch_idx in range(0, len(img_path_list), batch_size):
            path_set = img_path_list[batch_idx : batch_idx + batch_size]

            img_set = np.zeros((len(path_set),) + self.img_size)
            for img_idx, path in enumerate(path_set):
                img_set[img_idx] = self.__imread(path)
            img_set = img_set / 127.5 - 1
            yield img_set

#%%
class GAN:
    def __init__(self, noise_dim, img_size=(64, 64, 3)):
        self.noise_dim = noise_dim
        self.img_size = img_size
        self.dataloader = DataLoader('../../DL/cartoon/cartoon/*.png', self.img_size)
        
    def build_generator(self):     
        def conv_block(filter_num, h):
            h = Conv2D(filter_num, 3, padding='same')(h)
            h = UpSampling2D()(h)
            h = BatchNormalization(momentum=.8)(h)
            return LeakyReLU(alpha=.2)(h)
        
        noise_input = Input(shape=(self.noise_dim,))
        h = Dense(8*8*128, activation='selu')(noise_input)
        h = Reshape((8, 8, 128))(h)
        h = conv_block(128, h)
        h = conv_block(64, h)
        h = conv_block(32, h)
        h = Conv2D(self.img_size[-1], 3, activation='tanh', padding='same')(h)
        
        return Model(noise_input, h)
    
    def build_discriminator(self):
        img_input = Input(shape=self.img_size)
        h = Conv2D(64, 3, strides=2)(img_input)
        h = LeakyReLU(alpha=.2)(h)
        h = Conv2D(72, 3, strides=2)(h)
        h = LeakyReLU(alpha=.2)(h)
        h = Conv2D(96, 3, strides=2)(h)
        h = LeakyReLU(alpha=.2)(h)
        h = Conv2D(128, 3, strides=2)(h)
        h = LeakyReLU(alpha=.2)(h)
        h = Flatten()(h)
        h = Dropout(.4)(h)
        h = Dense(1, activation='sigmoid')(h)
        
        return Model(img_input, h)
    
    def __sample_image(self, epoch):
        r, c = 8, 8
        noise = np.random.standard_normal((r*c, self.noise_dim))
        img = self.generator.predict(noise).reshape((r, c) + self.img_size)
        img = img * .5 + .5
        fig = plt.figure(figsize=(20, 20))
        axs = fig.subplots(r, c)
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(img[i, j])
                axs[i, j].axis('off')
        fig.savefig('../Image/%d.png' % epoch)
        plt.close()
        
    def connect(self):
        self.generator = self.build_generator()
#         self.generator.summary()
        print(self.generator.count_params())
        self.discriminator = self.build_discriminator()
#         self.discriminator.summary()
        print(self.discriminator.count_params())
        
        self.d_optimizer = Adam(.0002, .5)
        self.g_optimizer = Adam(.0002, .5)
        self.discriminator.compile(optimizer=self.d_optimizer, loss='binary_crossentropy', metrics=['acc'])
        
        noise = Input(shape=(self.noise_dim,))
        img = self.generator(noise)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        
        self.combined = Model(noise, validity)
        self.combined.compile(optimizer=self.g_optimizer, loss='binary_crossentropy')
        
    def train(self, epochs, batch_size, sample_interval=10):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        self.history = []
        for e in range(epochs):
            for i, real_img in enumerate(self.dataloader.sampling_data(batch_size)):
                # Train D
                noise = np.random.standard_normal((batch_size, self.noise_dim))
                fake_img = self.generator.predict(noise)

                d_loss_real, real_acc = self.discriminator.train_on_batch(real_img, valid[:len(real_img)])
                d_loss_fake, fake_acc = self.discriminator.train_on_batch(fake_img, fake)
                d_loss = .5 * (d_loss_real + d_loss_fake)
                d_acc = .5 * (real_acc + fake_acc)

                noise = np.random.standard_normal((batch_size, self.noise_dim))
                g_loss = self.combined.train_on_batch(noise, valid)

                if i % sample_interval == 0:
                    info = {
                        'epoch': e,
                        'iter': i,
                        'd_loss': d_loss,
                        'd_acc': d_acc*100,
                        'g_loss': g_loss
                        }
                    self.history.append(list(info.values()))
                    print('[Epoch %(epoch)d][Iteration %(iter)d][D loss: %(d_loss).6f, acc: %(d_acc).2f%%][G loss: %(g_loss).6f]' % info)
            self.__sample_image(e)
        return self.history

#%%
gan = GAN(128, img_size=(64, 64, 3))
gan.connect()
gan.train(200, 256, sample_interval=10)