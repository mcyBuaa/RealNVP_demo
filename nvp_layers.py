# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:07:41 2022

@author: mcy
"""

from keras.layers import *
import numpy as np
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
import keras.initializers


class Permute(Layer):
    """排列层，提供两种方式重新排列最后一个维度的数据
    一种是直接反转，一种是随机打乱，默认是直接反转维度
    New Permute layer. Reverse or shuffle the final axis of inputs
    """
    def __init__(self, mode='reverse', **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.idxs = None # 打乱顺序的序id
        self.mode = mode
    def build(self, input_shape):
        super(Permute, self).build(input_shape)
        in_dim = input_shape[-1]
        if self.idxs is None:
            if self.mode == 'reverse':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(input_shape[-1],),
                                            dtype='int32',
                                            initializer=self.reverse_initializer,
                                            trainable=False)
            elif self.mode == 'random':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(input_shape[-1],),
                                            dtype='int32',
                                            initializer=self.random_initializer,
                                            trainable=False)
    def reverse_initializer(self, shape, dtype=None):
        idxs = list(range(shape[0]))
        return idxs[::-1]
    def random_initializer(self, shape, dtype=None):
        idxs = list(range(shape[0]))
        np.random.shuffle(idxs)
        return idxs
    def call(self, inputs):
        num_axis = K.ndim(inputs)
        inputs = K.permute_dimensions(inputs, list(range(num_axis))[::-1])
        x_outs = K.gather(inputs, self.idxs)
        x_outs = K.permute_dimensions(x_outs, list(range(num_axis))[::-1])
        return x_outs
    def inverse(self):
        in_dim = K.int_shape(self.idxs)[0]
        #序号排序之后的序号可以得到原序列
        reverse_idxs = tf.nn.top_k(self.idxs, in_dim)[1][::-1]
        layer = Permute()
        layer.idxs = reverse_idxs
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'idxs':self.idxs,
            'mode':self.mode,
            })
        return config


class Split(Layer):
    """将输入分区沿着最后一个轴为切分为若干部分
    pattern：切分模式，记录每一部分的大小的list；默认对半切分为两部分
    split inputs into several parts according pattern
    """
    def __init__(self, pattern=None, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.pattern = pattern
    def call(self, inputs):
        if self.pattern is None:
            in_dim = K.int_shape(inputs)[-1]
            self.pattern = [in_dim//2, in_dim - in_dim//2]
        partion = [0] + list(np.cumsum(self.pattern))
        return [inputs[..., i:j] for i,j in zip(partion, partion[1:])]
    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] + (d,) for d in self.pattern]
    def inverse(self):
        layer = Concat()
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pattern':self.pattern,
            })
        return config


class Concat(Layer):
    """把最后一个轴拼接起来
    like Concatenate but add inverse()
    """
    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)
    def call(self, inputs):
        self.pattern = [K.int_shape(i)[-1] for i in inputs]
        return K.concatenate(inputs, -1)
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (sum(self.pattern),)
    def inverse(self):
        layer = Split(self.pattern)
        return layer
    def get_config(self):
        config = super().get_config().copy()
        return config


class AffineCouple(Layer):
    """仿射耦合层
    """
    def __init__(self,
                 isinverse=False,
                 **kwargs):
        super(AffineCouple, self).__init__(**kwargs)
        self.isinverse = isinverse
    def call(self, inputs):
        """如果inputs的长度为3，那么就是加性耦合，否则就是一般的仿射耦合。
        if len(inputs) == 3, it equals additive coupling.
        if len(inputs) == 4, it is common affine coupling.
        """
        if len(inputs) == 3:
            x1, x2, shift = inputs
            log_scale = K.constant([0.])
        elif len(inputs) == 4:
            x1, x2, shift, log_scale = inputs
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0)) # 对数行列式
            x_outs = [x1, K.exp(-log_scale) * (x2 - shift)]
        else:
            logdet = -K.sum(K.mean(log_scale, 0)) # 对数行列式
            x_outs = [x1, K.exp(log_scale) * x2 + shift]
        #self.logdet = logdet
        self.add_loss(logdet)
        return x_outs
    def inverse(self):
        layer = AffineCouple(not self.isinverse)
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'isinverse':self.isinverse,
            })
        return config


class CoupleWrapper:
    """仿射耦合层的封装，使得可以直接将模型作为参数传入
    just a wrapper of AffineCouple for simpler use.
    """
    def __init__(self,
                 shift_model,
                 log_scale_model=None,
                 isinverse=False):
        self.shift_model = shift_model
        self.log_scale_model = log_scale_model
        self.layer = AffineCouple(isinverse)
    def __call__(self, inputs, whocare=0):
        x1, x2 = inputs
        shift = self.shift_model(x1)
        if whocare == 0:
            layer = self.layer
        else:
            layer = self.layer.inverse()
        if self.log_scale_model is None:
            return layer([x1, x2, shift])
        else:
            log_scale = self.log_scale_model(x1)
            return layer([x1, x2, shift, log_scale])
    def inverse(self):
        return lambda inputs: self(inputs, 1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shift_model':self.shift_model,
            'log_scale_model':self.log_scale_model,
            'layer':self.layer,
            })
        return config


class Actnorm(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self,
                 isinverse=False,
                 use_shift=True,
                 **kwargs):
        super(Actnorm, self).__init__(**kwargs)
        self.log_scale = None
        self.shift = None
        self.isinverse = isinverse
        self.use_shift = use_shift
    def build(self, input_shape):
        super(Actnorm, self).build(input_shape)
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)
        if self.log_scale is None:
            #init=keras.initializers.RandomUniform(minval=1e-5, maxval=1e-4)
            #待估计的方差项
            self.log_scale = self.add_weight(name='log_scale',
                                             shape=kernel_shape,
                                             initializer='zeros',
                                             trainable=True)
            '''
            扰动项，防止分母为0，可用常数或随机数。该项有进一步的优化空间，
            对于随机扰动，目前来看选择正态分布的结果好于均匀分布,
            但正态分布情况下，模型在训练初期仍然会产生loss为nan的问题
            '''
            '''
            self.noise = self.add_weight(name='noise',
                                             shape=kernel_shape,
                                             initializer='random_normal',
                                             trainable=False)
            '''
            self.noise = 1e-3
            std=K.sqrt(self.log_scale + self.noise)
            #保证方差大于0
            self.total=K.maximum(std, K.sqrt(K.constant(self.noise)))
            
        if self.use_shift and self.shift is None:
            #待估计的均值项
            self.shift = self.add_weight(name='shift',
                                         shape=kernel_shape,
                                         initializer='zeros',
                                         trainable=True)
        if not self.use_shift:
            self.shift = 0.
    def call(self, inputs):
        if self.isinverse:
            #logdet = K.sum(self.log_scale)
            logdet = K.sum(self.total)
            #x_outs = K.exp(-self.log_scale) * (inputs - self.shift)
            x_outs = K.exp(-self.total) * (inputs - self.shift)
        else:
            #logdet = -K.sum(self.log_scale)
            logdet = -K.sum(self.total)
            #x_outs = K.exp(self.log_scale) * inputs + self.shift
            x_outs = K.exp(self.total) * inputs + self.shift
        if K.ndim(inputs) > 2:
            #这一步的目的不清楚诶。。。
            logdet *= K.prod(K.cast(K.shape(inputs)[1:-1], 'float32'))
        #self.logdet = logdet
        self.add_loss(logdet)
        return x_outs
    def inverse(self):
        layer = Actnorm(not self.isinverse)
        layer.log_scale = self.log_scale
        layer.shift = self.shift
        layer.total = self.total
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'log_scale':self.log_scale,
            'shift':self.shift,
            'isinverse':self.isinverse,
            'use_shift':self.use_shift,
            'noise':self.noise,
            'total':self.total,
            })
        return config


class CondActnorm(Layer):
    """双输入缩放平移变换层（Conditional scale and shift）
    将x1做缩放平移，其中缩放平移量由x2算出来
    返回变换后的x1
    """
    def __init__(self,
                 isinverse=False,
                 use_shift=True,
                 **kwargs):
        super(CondActnorm, self).__init__(**kwargs)
        self.kernel = None
        self.bias = None
        self.isinverse = isinverse
        self.use_shift = use_shift
    def build(self, input_shape):
        super(CondActnorm, self).build(input_shape)
        in_dim = input_shape[0][-1]
        if self.use_shift:
            out_dim = in_dim * 2
        else:
            out_dim = in_dim
        if self.kernel is None:
            self.kernel = self.add_weight(name='kernel',
                                          shape=(3, 3, in_dim, out_dim),
                                          initializer='zeros',
                                          trainable=True)
        if self.bias is None:
            self.bias = self.add_weight(name='bias',
                                        shape=(out_dim,),
                                        initializer='zeros',
                                        trainable=True)
    def call(self, inputs):
        x1, x2 = inputs
        in_dim = K.int_shape(x1)[-1]
        x2_conv2d = K.conv2d(x2, self.kernel, padding='same')
        x2_conv2d = K.bias_add(x2_conv2d, self.bias)
        if self.use_shift:
            log_scale,shift = x2_conv2d[..., :in_dim], x2_conv2d[..., in_dim:]
        else:
            log_scale,shift = x2_conv2d, 0.
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(-log_scale) * (x1 - shift)
        else:
            logdet = -K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(log_scale) * x1 + shift
        #self.logdet = logdet
        self.add_loss(logdet)
        return x_outs
    def inverse(self):
        layer = CondActnorm(not self.isinverse)
        layer.kernel = self.kernel
        layer.bias = self.bias
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel':self.kernel,
            'bias':self.bias,
            'isinverse':self.isinverse,
            'use_shift':self.use_shift,
            })
        return config


class Reshape(Layer):
    """重新定义Reshape层，默认为Flatten
    主要目的是添加inverse方法
    combination of keras's Reshape and Flatten. And add inverse().
    """
    def __init__(self, shape=None, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape
    def call(self, inputs):
        self.in_shape = [i or -1 for i in K.int_shape(inputs)]
        if self.shape is None:
            self.shape = [-1, np.prod(self.in_shape[1:])]
        return K.reshape(inputs, self.shape)
    def compute_output_shape(self, input_shape):
        return tuple([i if i != -1 else None for i in self.shape])
    def inverse(self):
        return Reshape(self.in_shape)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape':self.shape,
            })
        return config


class Squeeze(Layer):
    """shape=[h, w, c] ==> shape=[h/n, w/n, n*n*c]
    """
    def __init__(self, factor=2, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.factor = factor
    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]
        assert height % self.factor == 0 and width % self.factor == 0
        #相当于在矩阵的行和列上分别新增了一个子维度，并在子维度上进行通道的加深
        inputs = K.reshape(inputs, (-1,
                                    height//self.factor,
                                    self.factor,
                                    width//self.factor,
                                    self.factor,
                                    channel))
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
        x_outs = K.reshape(inputs, (-1,
                                     height//self.factor,
                                     width//self.factor,
                                     channel*self.factor**2))
        return x_outs
    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return  (None, height//self.factor,
                 width//self.factor, channel*self.factor**2)
    def inverse(self):
        layer = UnSqueeze(self.factor)
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factor':self.factor,
            })
        return config


class UnSqueeze(Layer):
    """shape=[h, w, c] ==> shape=[h*n, w*n, c/(n*n)]
    """
    def __init__(self, factor=2, **kwargs):
        super(UnSqueeze, self).__init__(**kwargs)
        self.factor = factor
    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]
        assert channel % (self.factor**2) == 0
        inputs = K.reshape(inputs, (-1,
                                    height,
                                    width,
                                    self.factor,
                                    self.factor,
                                    channel//(self.factor**2)))
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
        x_outs = K.reshape(inputs, (-1,
                                     height*self.factor,
                                     width*self.factor,
                                     channel//(self.factor**2)))
        return x_outs
    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return  (None, height*self.factor,
                 width*self.factor, channel//(self.factor**2))
    def inverse(self):
        layer = Squeeze(self.factor)
        return layer
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factor':self.factor,
            })
        return config