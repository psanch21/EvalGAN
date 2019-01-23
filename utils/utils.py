import os
from scipy.io import loadmat
import numpy as np
import tensorflow as tf

import scipy
import math

import time

    
'''  ------------------------------------------------------------------------------
                                    OTHER
 ------------------------------------------------------------------------------ '''
    
    

def get_dataset(dataset, data_path, n_images, test):
        
    if(dataset=='cifar10'):
        tmp_file = 'data_batch_1' if not test else 'test_batch'
        filename = os.path.join(data_path, tmp_file)
        pickle = unpickle(filename)
        x_test = pickle[b'data'][:n_images] 
        min_value = x_test.min()
        x_test = x_test/127.5 -1 # Normalize between [-1,1]
        dims = [-1,3, 32, 32]
        dims_trans  = [0, 2, 3, 1]
        x_test = np.reshape(x_test, dims).transpose(dims_trans) 
        
        dim_img = [1, 32,32,3]
    
    return x_test, dim_img



'''  ------------------------------------------------------------------------------
                                    Distance metrics
 ------------------------------------------------------------------------------ '''
def angle(u,v):

    c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
    return np.arccos(np.clip(c, -1, 1))


def MSE(x_test, x_recons, axis=None):
    return np.mean((x_test-x_recons)**2, axis)
def PSNR(x_test, x_recons, axis=None):
    mse = MSE(x_test, x_recons, axis)
    return 10*np.log10(255**2/mse)
 
    
'''  ------------------------------------------------------------------------------
                                    FILES & DIRS
 ------------------------------------------------------------------------------ '''

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        my_dict = pickle.load(fo, encoding='bytes')
    return my_dict
         




def save_fig(f, name, latex=False):
    if(latex):
        name = name.replace('.png', '.tex')
        tikz_save(name)
    else:
        f.savefig(name,bbox_inches='tight', pad_inches=0)
        
        
        
def plot_hist(x, ax, xlabel='', title='',color='darkblue', label=None):

    
    ax.hist(x,label=label,color=color, alpha=.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax
    

def plot_scatter(data, ax, label=None, title=''):
    
    ax.grid(True)

    ax.scatter(data[0], data[1],label=label, alpha=0.6)
    
    ax.set_xlabel('PSNR')
    ax.set_ylabel('$\log_{10} p(x) +Z $')
    ax.set_title(title)
    if(label is not None):
        ax.legend()
    return ax