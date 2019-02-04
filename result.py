import os, sys
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")
import os
import utils.utils as utils


'''  ------------------------------------------------------------------------------
                                     GET DATA
    ------------------------------------------------------------------------------ '''
x_test, dim_img = utils.get_dataset(dataset='cifar10', data_path='./data', n_images=50, test=1)



from evalGAN import EvalGAN


evalGAN_SN = EvalGAN(output_dir='./cifar10', 
                  model_name='SNDCGAN')
evalGAN_SN.set_data(x_test)

evalGAN_GP = EvalGAN(output_dir='./cifar10', 
                  model_name='WGANGP')
evalGAN_GP.set_data(x_test)


evalGAN_SN.print_results()
f = evalGAN_SN.plot()

f2 = evalGAN_GP.plot()

utils.save_fig(f,'evalGAN_c10SN.png', latex=False)
utils.save_fig(f2,'evalGAN_c10GP.png', latex=False)



f = plt.figure()
ax = plt.subplot(1,1,1)
data_SN = evalGAN_SN.get_scatter_data()
utils.plot_scatter(data_SN, ax, label=evalGAN_SN.model_name)

data_GP = evalGAN_GP.get_scatter_data()
utils.plot_scatter(data_GP, ax, label=evalGAN_GP.model_name)

utils.save_fig(f,'evalGAN_scatter.png', latex=False)
