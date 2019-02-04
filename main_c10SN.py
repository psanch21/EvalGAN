import tensorflow as tf
import os, sys
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils.utils as utils


'''  ------------------------------------------------------------------------------
                                     GET DATA
    ------------------------------------------------------------------------------ '''

x_test, _ = utils.get_dataset(dataset='cifar10', data_path='./data', n_images=50, test=1)



from evalGAN import EvalGAN

z_dim=256

evalGAN = EvalGAN(input_G_name='z',
                  output_G_name='generator/out:0',
                  batch_size=64,
                  z_dim=z_dim,
                  z_init=tf.initializers.random_normal(0.,1.),
                  constraint=lambda t: tf.clip_by_norm(t,  np.sqrt(z_dim)),
                  learning_rate=5e-3,
                  beta1=0.9,
                  checkpoint_dir='./c10_SN', 
                  output_dir='./cifar10', 
                  model_name='SNDCGAN')

evalGAN.set_data(x_test)
evalGAN.add_placeholder('Placeholder', False)

evalGAN.fit(epochs=2000, early_stopping=1, restore=1)
sigma_list = np.linspace(0.001,0.2,1000)
evalGAN.analysis_isotropic(sigma_list, N=1000)
evalGAN.analysis_non_isotropic(sigma_list, N=1000,N_max=100000, T=40)


