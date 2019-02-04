import tensorflow as tf
import os, sys
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.close("all")
import os
import utils.utils as utils



# Here you get load your data in an array with shape [n_samples, ...]
# x_test = 



from evalGAN import EvalGAN

''' 
Examples of constraint param

constraint = None
constraint = lambda t: tf.clip_by_norm(t,  np.sqrt(z_dim))
constraint = lambda t: tf.clip_by_value(t, -1, 1)

Examples of z_init param

z_init=tf.initializers.random_normal(0.,1.)
z_init=tf.initializers.random_uniform(-1.,1.)

'''

z_dim=256

evalGAN = EvalGAN(input_G_name='z',
                  output_G_name='generator/out:0',
                  batch_size=64,
                  z_dim=z_dim,
                  z_init=tf.initializers.random_normal(0.,1.),
                  constraint=lambda t: tf.clip_by_norm(t,  np.sqrt(z_dim)),
                  learning_rate=5e-3,
                  beta1=0.9,
                  checkpoint_dir='./checkpoint_folder', 
                  output_dir='./result_folder', 
                  model_name='GAN')

# Fix the data for evaluation
evalGAN.set_data(x_test)

'''
Add any placeholder your TensorFlow graph needs. Add them one by one

evalGAN.add_placeholder('Placeholder1', False)
evalGAN.add_placeholder('Placeholder2', 0.5)
'''
evalGAN.add_placeholder('Placeholder', False)

# Optimaze image by image to get optimal reconstruction and input noise vector
evalGAN.fit(epochs=3, early_stopping=1, restore=1)

# Select a list of sigma_e to approximate marginal likelihood
sigma_list = np.linspace(0.001,0.2,100)

evalGAN.analysis_isotropic(sigma_list, N=128)
evalGAN.analysis_non_isotropic( sigma_list, N=1000,N_max=10000, T=40)

# Plot EvalGAN results
f = evalGAN_SN.plot()


