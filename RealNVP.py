# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:06:07 2022

@author: mcy
"""

#Real NVP模型，根据苏神的Glow模型代码进行了适当修改
#参考文章https://kexue.fm/archives/5807

from keras.layers import *
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import Callback
from keras.optimizers import adam_v2
from nvp_layers import *
import imageio
import numpy as np
import glob
import os
from PIL import Image
from keras.models import load_model

'''
#对cifar10数据集进行训练，但是100个epoch内的效果不太好
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255 * 2 - 1
x_test = x_test.astype('float32') / 255 * 2 - 1
'''

#选择CelebaA数据集进行训练(从中随机选取6万张图片)，图片大小32*32*3
imgs = glob.glob('./CelebA_32_sample/*.jpg')
img_size = 32 #x_train.shape[1]  #图像尺寸
depth = 10  # 单步flow运算内部的深度
level = 3  # flow模块数量

def imread(f):
    x = Image.open(f)
    x = np.array(x)
    x = x.astype(np.float32)
    return x / 255 * 2 - 1

#由于图像过多，采用生成器逐步读入
def data_generator(batch_size=64):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X,X.reshape((X.shape[0], -1))
                X = []


def build_basic_model(in_channel):
    """用于拟合耦合层中s和t函数的卷积神经网络 
    """
    _in = Input(shape=(None, None, in_channel))
    _ = _in
    hidden_dim = 512
    _ = Conv2D(hidden_dim,
               (3, 3),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(hidden_dim,
               (1, 1),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('relu')(_)
    _ = Conv2D(in_channel,
               (3, 3),
               kernel_initializer='zeros',
               padding='same')(_)
    return Model(_in, _)


#模型squeeze操作的初始化
squeeze = Squeeze()
#inner_layers用于保存flow运算内部各层的变换过程
#outer_layers用于保存flow模块外的变换过程，二者都是为了之后更好的求逆
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])

for i in range(3):
    outer_layers.append([])


x_in = Input(shape=(img_size, img_size, 3))
x = x_in
x_outs = []  #用于分段存储输入

# 给输入加入噪声（add noise into inputs for stability.）
x = Lambda(lambda s: K.in_train_phase(s + 1./256 * K.random_uniform(K.shape(s)), s))(x)

#RealNVP的模型结构
for i in range(level):
    x = squeeze(x)  #squeeze操作增加通道数
    for j in range(depth):
        actnorm = Actnorm()  #代替banch normalization完成尺度变换
        permute = Permute(mode='random')  #打乱通道
        split = Split()  #沿通道轴拆分图像
        #仿射耦合层，由于squeeze操作，s和t的输入通道数随外循环成倍增加
        couple = CoupleWrapper(build_basic_model(3*2**(i+1)),build_basic_model(3*2**(i+1)))  
        concat = Concat()  #沿通道轴拼接图像
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)
        x = permute(x)
        x1, x2 = split(x)
        x1, x2 = couple([x1, x2])
        x = concat([x1, x2])
    if i < level-1:
        split = Split()
        condactnorm = CondActnorm() #外层的多尺度结构，基于一半输入计算另一半的条件分布
        reshape = Reshape()  #用于拉直矩阵
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()  #最后一个flow输出不做分割，分布独立，故用普通的多尺度变换
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])

encoder = Model(x_in, x)

encoder.summary()
'''
#目标(损失)函数：输入图像分布q(x)的最大似然(整体取负用adam优化)，以下为第一部分，即-0.5||f(x)||^2和常数项，
雅各比行列式在尺度变换层和仿射耦合层已经加入到目标函数中
'''
encoder.compile(loss=lambda y_true,y_pred: 0.5 * K.sum(y_pred**2, 1) + 0.5 * np.log(2*np.pi) * K.int_shape(y_pred)[1],
                optimizer=adam_v2.Adam(1e-5))


# 搭建逆模型（生成模型），将所有操作倒过来执行

x_in = Input(shape=K.int_shape(encoder.outputs[0])[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x


for i,(split,condactnorm,reshape) in enumerate(list(zip(*outer_layers))[::-1]):
    if i > 0:
        x1 = x
        x_out = outputs[-i]
        x_out = reshape.inverse()(x_out)
        x2 = condactnorm.inverse()([x_out, x1])
        x = split.inverse()([x1, x2])
    for j,(actnorm,permute,split,couple,concat) in enumerate(list(zip(*inner_layers))[::-1][i*depth: (i+1)*depth]):
        x1, x2 = concat.inverse()(x)
        x1, x2 = couple.inverse()([x1, x2])
        x = split.inverse()([x1, x2])
        x = permute.inverse()(x)
        x = actnorm.inverse()(x)
    x = squeeze.inverse()(x)


decoder = Model(x_in, x)

def sample(path, std=1):
    """采样查看生成效果（generate samples per epoch）
    """
    n = 9 #随机生成n**2张人脸
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            decoder_input_shape = (1,) + K.int_shape(decoder.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size: (i + 1) * img_size,
                   j * img_size: (j + 1) * img_size] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255).astype('uint8')
    imageio.imwrite(path, figure)
    

#模型训练时的回调函数，用于每轮图像生成效果的记录以及模型最优参数的保存
class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        global encoder
        path = './test_pic/test_%s.png' % epoch
        sample(path, 0.9)
        if logs['loss'] <= self.lowest:
            #记录损失最小的模型参数
            self.lowest = logs['loss']
            encoder.save_weights('./best_model/best_encoder.weights')
            #encoder.save('./best_model/model150.h5')
        elif logs['loss'] > 0 and epoch > 10:
            """在后面，loss一般为负数，一旦重新变成正数，
            就意味着模型已经崩溃，需要降低学习率。
            In general, loss is less than zero.
            If loss is greater than zero again, it means model has collapsed.
            We need to reload the best model and lower learning rate.
            """
            encoder.load_weights('./best_model/best_encoder.weights')
            #encoder.load_weights('./best_model/model150.h5')
            K.set_value(encoder.optimizer.lr, 1e-5)


evaluator = Evaluate()

#训练模型并进行样本生成

#encoder.load_weights('./best_model/best_encoder.weights')

encoder.fit_generator(data_generator(),
                      steps_per_epoch=500,
                      epochs=200,
                      callbacks=[evaluator])



'''
encoder.fit(x_train,
            x_train,
            batch_size=64,
            epochs=100,
            validation_data=(x_test, x_test),
            callbacks=[evaluator])
'''
#sample('test.png',0.9)

#%%模型权重的载入

#encoder模型初始化
#模型squeeze操作的初始化
squeeze = Squeeze()
#inner_layers用于保存flow运算内部各层的变换过程
#outer_layers用于保存flow模块外的变换过程，二者都是为了之后更好的求逆
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])

for i in range(3):
    outer_layers.append([])


x_in = Input(shape=(img_size, img_size, 3))
x = x_in
x_outs = []  #用于分段存储输入

# 给输入加入噪声（add noise into inputs for stability.）
x = Lambda(lambda s: K.in_train_phase(s + 1./256 * K.random_uniform(K.shape(s)), s))(x)

#RealNVP的模型结构
for i in range(level):
    x = squeeze(x)  #squeeze操作增加通道数
    for j in range(depth):
        actnorm = Actnorm()  #代替banch normalization完成尺度变换
        permute = Permute(mode='random')  #打乱通道
        split = Split()  #沿通道轴拆分图像
        #仿射耦合层，由于squeeze操作，s和t的输入通道数随外循环成倍增加
        couple = CoupleWrapper(build_basic_model(3*2**(i+1)),build_basic_model(3*2**(i+1)))  
        concat = Concat()  #沿通道轴拼接图像
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)
        x = permute(x)
        x1, x2 = split(x)
        x1, x2 = couple([x1, x2])
        x = concat([x1, x2])
    if i < level-1:
        split = Split()
        condactnorm = CondActnorm() #外层的多尺度结构，基于一半输入计算另一半的条件分布
        reshape = Reshape()  #用于拉直矩阵
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()  #最后一个flow输出不做分割，分布独立，故用普通的多尺度变换
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])

encoder1 = Model(x_in, x)

#encoder先载入权重
encoder1.load_weights('./best_model/best_encoder.weights')

#decoder模型搭建
x_in = Input(shape=K.int_shape(encoder1.outputs[0])[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x


for i,(split,condactnorm,reshape) in enumerate(list(zip(*outer_layers))[::-1]):
    if i > 0:
        x1 = x
        x_out = outputs[-i]
        x_out = reshape.inverse()(x_out)
        x2 = condactnorm.inverse()([x_out, x1])
        x = split.inverse()([x1, x2])
    for j,(actnorm,permute,split,couple,concat) in enumerate(list(zip(*inner_layers))[::-1][i*depth: (i+1)*depth]):
        x1, x2 = concat.inverse()(x)
        x1, x2 = couple.inverse()([x1, x2])
        x = split.inverse()([x1, x2])
        x = permute.inverse()(x)
        x = actnorm.inverse()(x)
    x = squeeze.inverse()(x)


decoder1 = Model(x_in, x)

def sample1(path, std=1):
    """采样查看生成效果（generate samples per epoch）
    """
    n = 9 #随机生成n**2张人脸
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            decoder_input_shape = (1,) + K.int_shape(decoder1.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
            x_decoded = decoder1.predict(z_sample)
            digit = x_decoded[0].reshape(img_size, img_size, 3)
            figure[i * img_size: (i + 1) * img_size,
                   j * img_size: (j + 1) * img_size] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.clip(figure, 0, 255).astype('uint8')
    imageio.imwrite(path, figure)

#模型测试
sample1('test1.png',0.9)