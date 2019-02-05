import tensorflow as tf
import numpy as np

from utils.early_stopping import EarlyStopping
import sys
import os
import utils.utils as utils
import math
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class EvalGAN():
    def __init__(self,input_G_name='z', output_G_name='out_gen:0', batch_size=64,z_dim=256,
                  z_init=tf.initializers.random_normal(0.,1.),
                  constraint=None,
                  learning_rate=5e-3,
                  beta1=0.9,
                  checkpoint_dir=None,
                  output_dir='./recons',
                  model_name='GAN'):

        self.input_G_name = input_G_name
        self.output_G_name = 'import/' + output_G_name
        self.bs = batch_size
        self.z_dim = z_dim
        self.z_init = z_init
        self.constraint = constraint
        self.l_rate = learning_rate
        self.beta1 = beta1
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if(checkpoint_dir is not None):
            self.latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.meta_file = self.latest_checkpoint + '.meta'

        self.model_name = model_name
        # self.evalGAN_config = os.path.join(output_dir, '{}_config.npz'.format(model_name))
        self.evalGAN_recons = os.path.join(output_dir, '{}_recons.npz'.format(model_name))
        self.evalGAN_iso = os.path.join(output_dir, '{}_iso.npz'.format(model_name))
        self.evalGAN_non_iso = os.path.join(output_dir, '{}_non_iso.npz'.format(model_name))


        self.datum_dims = None

        self.placeholder_dict = dict()

    '''
    ------------------------------------------------------------------------------
                                         DATA METHODS
    ------------------------------------------------------------------------------
    '''
    def normalize_data(self, data):
        return data

    def denormalize_data(self, data):
        if(data.max()< 1.01 and data.min() < -0.01):
            data =  (data+1.)*127.5
        elif(data.max()< 1.01 and data.min() > -0.01):
            data =  (data)*255.

        if(data.shape[-1]>3):
            data = np.transpose(data,(0,2,3,1))
        return data.astype(int)

    def dist(self, x_test, x_recons):
        mse = np.mean((x_test-x_recons)**2, (1,2,3))
        return 10*np.log10(255**2/mse)

    def set_data(self, x_test):

        self.x_test = self.normalize_data(x_test)
        self.n_imgs = x_test.shape[0]

    def get_z_dim(self):
        evalGAN_data = self.get_evalGAN_recons()
        return evalGAN_data['z_infer'].shape[-1]

    def get_recons_data(self):
        try:
            evalGAN_data = np.load(self.evalGAN_recons)
        except:
            evalGAN_data = dict()
            evalGAN_data['z_infer'] = list()
            evalGAN_data['x_recons'] = list()
            print('EvalGAN recons data not found')
        return evalGAN_data['z_infer'], evalGAN_data['x_recons']

    def get_evalGAN_recons(self):
        print('EvalGAN Recons Data: {}'.format(self.evalGAN_recons))
        evalGAN_data = None
        try:
            evalGAN_data = np.load(self.evalGAN_recons)
        except:
            print('EvalGAN recons data not found')
        return evalGAN_data


    def get_iso_data(self):
        try:
            evalGAN_data = np.load(self.evalGAN_iso)
        except:
            evalGAN_data = dict()
            evalGAN_data['mse'] = list()
            evalGAN_data['psnr'] = list()
            evalGAN_data['sigma_list'] = list()
            print('EvalGAN iso data not found')
        return evalGAN_data['mse'], evalGAN_data['psnr'], evalGAN_data['sigma_list']

    def get_evalGAN_iso(self):
        try:
            evalGAN_data = np.load(self.evalGAN_iso)
        except:
            evalGAN_data = dict()
            evalGAN_data['mse'] = list()
            evalGAN_data['psnr'] = list()
            print('EvalGAN iso data not found')
        return evalGAN_data


    def get_evalGAN_non_iso(self):
        evalGAN_data = None
        try:
            evalGAN_data = np.load(self.evalGAN_non_iso)
        except:
            print('EvalGAN non-iso data not found')
        return evalGAN_data

    def add_placeholder(self, key, value):
        self.placeholder_dict[key] = value



    '''
    ------------------------------------------------------------------------------
                                         GRAPH
    ------------------------------------------------------------------------------
    '''

    def __create_inputs(self):
        with tf.variable_scope('inputs'):
            self.x_real = tf.placeholder(tf.float32,[None]+ self.datum_dims, name='x_real')


    def __create_graph(self,meta_file = None):
        self.z = tf.get_variable('z',
                                initializer=self.z_init,
                                shape=[1,self.z_dim],
                                trainable=True,
                                dtype=tf.float32,
                                constraint=self.constraint)


        self.z_tile = tf.tile(self.z,[self.bs,1])
        input_map = self.placeholder_dict.copy()
        input_map[self.input_G_name] = self.z_tile

        print('Meta file: {}'.format(meta_file))
        print('[*] Layer (', self.z.name, ') output shape:', self.z.get_shape().as_list())
        new_saver = tf.train.import_meta_graph(meta_file,
                                               input_map=input_map,
                                               import_scope='import')
        self.x_recons = tf.get_default_graph().get_tensor_by_name(self.output_G_name)
        self.x_recons = tf.unstack(self.x_recons, num=self.bs, axis=0)[0]
        print(self.x_recons.get_shape().as_list())
        return new_saver

    def __create_loss_optimizer(self):
        print('[*] Defining Loss Function and Optimizer...')
        self.recons_loss = self.loss_function(self.x_real, self.x_recons)

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.l_rate, beta1=self.beta1)
            self.train_step = self.optim.minimize(self.recons_loss, var_list=[self.z])


    def loss_function(self, x_real, x_recons):
        return tf.reduce_mean(tf.reduce_sum(tf.square(x_real - x_recons),[1,2,3]))
    '''  ------------------------------------------------------------------------------
                                         OPTIMIZATION
        ------------------------------------------------------------------------------ '''
    def __train_epoch(self, session, image):
        r_loss = []

        for _ in range(100):
            batch_x = image

            loss_dict = self.partial_fit(session,batch_x)

            r_loss.append(loss_dict['r_loss'])

        dict_loss = dict()
        dict_loss['r_loss'] = np.mean(r_loss)

        return dict_loss

    def train(self, image, epochs, enable_es=1):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            tf.set_random_seed(1234)

            self.__create_inputs()
            new_saver = self.__create_graph(self.meta_file)
            self.__create_loss_optimizer()

            # slim.model_analyzer.analyze_vars(tf.trainable_variables() , print_info=True)

            early_stopping = EarlyStopping(patience=30, min_delta=1e-1)

            tf.global_variables_initializer().run()

            new_saver.restore(session,self.latest_checkpoint)
            
            recons_loss = list()
            print('Starting optimization...')
            for cur_epoch in range(epochs + 1):


                dict_loss = self.__train_epoch(session,image)
                list_loss = list(dict_loss.values())

                if np.isnan(list_loss[0]):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

                if(cur_epoch % 20 == 0 or cur_epoch==0):
                    print('EPOCH: {} | dist: {} '.format(cur_epoch, list_loss[0]))
                    
                recons_loss.append(list_loss[0])
                #Early stopping
                if(cur_epoch>50 and enable_es==1 and early_stopping.stop(list_loss[0])):
                    print('Early Stopping!')
                    print('EPOCH: {} | dist: {} '.format(cur_epoch, list_loss[0]))
                    break


            z_infer =  session.run(self.z)
            x_recons = session.run(self.x_recons)

        return z_infer, x_recons, recons_loss



    def partial_fit(self, session, x):
        tensors = [self.train_step,
                   self.recons_loss]
        feed_dict = { self.x_real: x}
        _, recons_loss = session.run(tensors, feed_dict=feed_dict)

        loss_dict = dict()
        loss_dict['r_loss'] = recons_loss

        return loss_dict


    def fit(self, epochs=100, early_stopping=0,  restore=1):


        self.datum_dims=list(self.x_test.shape[1:])
        N = self.x_test.shape[0]
        z_infer = list()
        x_recons = list()
        loss_recons = list()
        init_n = 0
        if(restore == 1):
            evalGAN_data = self.get_evalGAN_recons()
            if(evalGAN_data is not None):
                z_infer = list(evalGAN_data['z_infer'])
                x_recons = list(evalGAN_data['x_recons'])
                loss_recons = list(evalGAN_data['loss_recons'])
                init_n = len(z_infer)
            else:
                z_infer = list()
                x_recons = list()
                loss_recons = list()
                init_n = 0

        for n in range(init_n, N):
            test_sample = self.x_test[n].reshape([1]+self.datum_dims)
            print('Datum {} / {}'.format(n,N))
            z_in, x_rec, loss = self.train(test_sample, epochs, early_stopping)
            z_infer.append(z_in)
            x_recons.append(x_rec)
            loss_recons.append(loss)
            if(n % 10 == 0):
                np.savez(self.evalGAN_recons, x_recons=np.array(x_recons),
                         z_infer=np.array(z_infer),
                         loss_recons=loss_recons)
        np.savez(self.evalGAN_recons,  x_recons=np.array(x_recons),
                 z_infer=np.array(z_infer),
                 loss_recons=loss_recons)



    '''
    ------------------------------------------------------------------------------
                                         ANALYSIS ISOTROPIC
    ------------------------------------------------------------------------------
    '''

    def __compute_isotropic(self, graph, sess,x_recons, z_infer, N, sigma_list):
        n_imgs = z_infer.shape[0]
        n_iter = math.ceil(N/self.bs)
        N = n_iter*self.bs
        n_sigma=len(sigma_list)
        mse_array = np.zeros([n_imgs,  n_sigma])
        psnr_array = np.zeros([n_imgs, n_sigma])

        n_init = 0
        print('Starting analysis isotropic | n_imgs: {}'.format(n_imgs))
        evalGAN_iso = self.get_evalGAN_iso()
        if(evalGAN_iso != None):
            mse_a = evalGAN_iso['mse']
            psnr_a = evalGAN_iso['psnr']
            sigma_list = evalGAN_iso['sigma_list']
            N = evalGAN_iso['N']
            n_sigma=len(sigma_list)
            n_init = mse_a.shape[0]

            print('Loading analysis isotropic | n_imgs: {}'.format(n_init))
            mse_array[:n_init,:] = mse_a
            psnr_array[:n_init,:] = psnr_a

        for i in range(n_init, n_imgs):
            for j in range(n_sigma):
                mse_tmp = np.zeros(N)
                psnr_tmp = np.zeros(N)
                for n in range(n_iter):
                    x_tmp = np.tile(x_recons[i], [self.bs,1,1,1])
                    z_tmp = np.tile(z_infer[i,:],[self.bs,1]) + np.random.normal(0,
                                   sigma_list[j],[self.bs, self.z_dim])
                    x_gener = graph.get_recons(sess, z_tmp)
                    x_gener = self.denormalize_data(x_gener)
                    mse_tmp[n*self.bs:(n+1)*self.bs] = utils.MSE(x_gener, x_tmp, (1,2,3))
                    psnr_tmp[n*self.bs:(n+1)*self.bs] = self.dist(x_gener, x_tmp)

                mse_array[i,j] = np.mean(mse_tmp)
                psnr_array[i,j] = np.mean(psnr_tmp)

                print('DATUM({}/{}) | MSE={} | PSNR: {}'.format(i,n_imgs, np.around(mse_array[i,j],4),
                             np.around(psnr_array[i,j],4)))
            print('')
        mse = mse_array
        psnr = psnr_array
        np.savez(self.evalGAN_iso, mse=mse,psnr=psnr, sigma_list=sigma_list, N=N)
        print('Analysis isotropic completed | n_imgs: {}'.format(n_init))
        return

    def analysis_isotropic(self, sigma_list, N=1000):
        evalGAN_data = self.get_evalGAN_recons()
        z_infer = evalGAN_data['z_infer']
        x_recons = evalGAN_data['x_recons']
        x_recons = self.denormalize_data(x_recons)
        gener_graph = GeneratorGraph(self.z_dim, self.input_G_name,
                                     self.output_G_name, self.placeholder_dict)
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            saver = gener_graph.create_graph(meta_file= self.meta_file)
            tf.global_variables_initializer().run()
            saver.restore(session,self.latest_checkpoint)

            self.__compute_isotropic(gener_graph, session,x_recons, z_infer, N, sigma_list)



    def get_sigma_bar(self, sigma_list, PSNR_array, T):
        N = PSNR_array.shape[0]
        sigma_bar = np.zeros(N)
        for i in range(N):
            f = interp1d(PSNR_array[i,:], sigma_list)
            sigma_bar[i] = f(T)
        return sigma_bar

    '''
    ------------------------------------------------------------------------------
                                 ANALYSIS NON ISOTROPIC
    ------------------------------------------------------------------------------
    '''
    def __compute_non_isotropic(self, graph, sess,x_recons, z_infer, N, N_max, T, sigma_list):
        n_imgs = z_infer.shape[0]
        n_iter = math.ceil(N/self.bs)
        N = n_iter*self.bs
        print('Starting analysis non-isotropic | n_imgs: {}'.format(n_imgs))
        evalGAN_non_iso = self.get_evalGAN_non_iso()
        if(evalGAN_non_iso != None):
            Nc_m = evalGAN_non_iso['Nc_matrix']
            P1_m = evalGAN_non_iso['P1_matrix']
            N_l = evalGAN_non_iso['N_list']
            T = evalGAN_non_iso['T']

            n = Nc_m.shape[0]

            n_sigma=len(sigma_list)
            sigma_list = evalGAN_non_iso['sigma_list']

            N_list = N*np.ones(n_imgs).astype(int)
            Nc_matrix = -np.ones([n_imgs, n_sigma])
            P1_matrix = np.zeros([n_imgs, n_sigma])

            print('Loading analysis non-isotropic | n_imgs: {}'.format(n))
            Nc_matrix[:n,:] = Nc_m
            P1_matrix[:n,:] = P1_m
            N_list[:n] = N_l
        else:

            N_list = N*np.ones(n_imgs).astype(int)
            n_sigma=len(sigma_list)
            Nc_matrix = -np.ones([n_imgs, n_sigma])
            P1_matrix = np.zeros([n_imgs, n_sigma])


        init_time = time.time()
        for i in range(n_imgs):
            cur_time = time.time()
            init_sigma = np.argmin(Nc_matrix[i,:])
            for j in range(init_sigma,n_sigma):
                N = N_list[i]
                if(N >=N_max):
                    break
                n_iter = math.ceil(N/self.bs)
                sigma = sigma_list[j]
                psnr_tmp = np.zeros(N)
                for n in range(n_iter):
                    x_tmp = np.tile(x_recons[i], [self.bs,1,1,1])
                    z_tmp = np.tile(z_infer[i,:],[self.bs,1]) + np.random.normal(0,
                                   sigma,[self.bs, self.z_dim])
                    x_gener = graph.get_recons(sess, z_tmp)
                    x_gener = self.denormalize_data(x_gener)
                    psnr_tmp[n*self.bs:(n+1)*self.bs] = self.dist(x_gener, x_tmp)

                Nc= np.sum(psnr_tmp > T)
                Nc_matrix[i,j] = Nc
                P1_approx = Nc/N
                P1_matrix[i,j] = P1_approx
                if(Nc <= 100):
                    N_p = N*10
                    n_iter = math.ceil(N_p/self.bs)
                    N_list[i] = n_iter*self.bs


                print('{}/{}| {} min | sigma={}, DATUM({}/{}) | N={} , Nc={} | P1: {}'.format(
                        j,n_sigma, int((time.time()-cur_time)/60), np.around(sigma_list[j],4),
                        i,n_imgs, N, Nc, P1_approx))
            if(i % 5 == 0):
                print('Saving...')
                np.savez(self.evalGAN_non_iso, N_list=N_list, sigma_list=sigma_list, Nc_matrix=Nc_matrix,
                         P1_matrix=P1_matrix, T=T)

            time_total = time.time()- init_time
            time_epoch = time.time()-cur_time
            print('\n Time: {} | Total time: {} \n'.format(int(time_epoch/60),int(time_total/60)))
            print('')
        np.savez(self.evalGAN_non_iso, N_list=N_list, sigma_list=sigma_list,
                 Nc_matrix=Nc_matrix, P1_matrix=P1_matrix, T=T)
        print('Analysis non-isotropic completed | n_imgs: {}'.format(n_imgs))
        return

    def analysis_non_isotropic(self, sigma_list, N=1000,N_max=10000, T=40):
        """Get data to compute non isotropic approximation of marginal loglikelihood.

        Usage::

            >>> import klassify
            >>> data = [("green", "foo"), ("orange", "bar")]
            >>> classifier = klassify.train(data)

        :param sigma_list: A list of increasing sigma_e.
        :param N: A list of tuples of the form ``(color, label)``.

        :return A :class:`Classifier <Classifier>`
        """
        evalGAN_data = self.get_evalGAN_recons()
        z_infer = evalGAN_data['z_infer']
        x_recons = evalGAN_data['x_recons']
        x_recons = self.denormalize_data(x_recons)
        gener_graph = GeneratorGraph(self.z_dim, self.input_G_name,
                                     self.output_G_name, self.placeholder_dict)
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            saver = gener_graph.create_graph(meta_file= self.meta_file)
            tf.global_variables_initializer().run()
            saver.restore(session,self.latest_checkpoint)

            self.__compute_non_isotropic(gener_graph, session,x_recons, z_infer, N, N_max, T, sigma_list)

        return

    '''
    ------------------------------------------------------------------------------
                                        GET RESULT DATA
    ------------------------------------------------------------------------------
    '''
    def __get_recons_PSNR(self):
        z_infer, x_recons = self.get_recons_data()
        if(x_recons.shape[0] != self.n_imgs):
            print('ERROR: Mismatch in number of samples |  n_recons: {} n_real: {}'
                  .format(x_recons.shape[0], self.n_imgs))
            return

        x_recons = self.denormalize_data(x_recons)
        self.x_test = self.denormalize_data(self.x_test)

        return self.dist(x_recons, self.x_test)

    def __get_non_iso_log_prob(self):
        evalGAN_non_iso_data = self.get_evalGAN_non_iso()
        Nc_matrix = evalGAN_non_iso_data['Nc_matrix']
        P1_matrix = evalGAN_non_iso_data['P1_matrix']
        sigma_list = evalGAN_non_iso_data['sigma_list']

        N = Nc_matrix.shape[0]

        if(N != self.n_imgs):
            print('ERROR Dimension mismatch | Nc_matrix: {} x_test: {}'
                  .format(N, self.n_imgs))
            return

        idx_sigma = np.sum(Nc_matrix > 0, 1)-1
        sigma_bar_non_iso = sigma_list[idx_sigma]
        z_dim = self.get_z_dim()
        log_prob = np.zeros(N)
        for i, idx in enumerate(idx_sigma):
            log_prob[i] = np.log10(P1_matrix[i,idx]) + z_dim*np.log10(sigma_bar_non_iso[i])

        return log_prob


    def get_scatter_data(self):
        psnr = self.__get_recons_PSNR()
        logprob = self.__get_non_iso_log_prob()
        return [psnr, logprob]


    def print_results(self):
        psnr = self.__get_recons_PSNR()
        logprob = self.__get_non_iso_log_prob()

        psnr = np.around(psnr, 1)
        logprob = np.around(logprob, 2)
        N = len(psnr)
        print('              {}                '.format(self.model_name))
        print('--------------------------------')
        print('     Samples   |  PSNR  |  LL(x)')
        for n in range(N):

            print('Sample({}/{})  |  {}  |  {}'.format(n,N, psnr[n], logprob[n]))


    '''
    ------------------------------------------------------------------------------
                                    PLOTS
    ------------------------------------------------------------------------------
    '''


    def plot_iso(self,data, ax, ylabel, label=None, color= 'darkblue', title=''):

        ax.grid(True)

        for i in range(data.shape[0]):
            if(i == 0):
                ax.plot(data[i,:], label=label, color=color)
            else:
                ax.plot(data[i,:], color=color)


        ax.set_xlabel('$\sigma_e$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if(label is not None):
            ax.legend()
        return ax


    def plot(self):
        psnr = self.__get_recons_PSNR()
        logprob = self.__get_non_iso_log_prob()
        mse_iso, psnr_iso, sigma_list = self.get_iso_data()

        f = plt.figure(figsize=(12,12))
        n_cols = 2
        n_rows = 2
        ax = plt.subplot(n_rows,n_cols,1)
        self.plot_iso(psnr_iso, ax, ylabel='PSNR')

        ax = plt.subplot(n_rows,n_cols,2)
        sigma_bar = self.get_sigma_bar(sigma_list, psnr_iso, T=40)
        utils.plot_hist(sigma_bar, ax, xlabel='$\sigma_e$', title='PSNR > 40')


        ax = plt.subplot(n_rows,n_cols,3)
        utils.plot_hist(logprob, ax, xlabel='$\log_{10} p(x) +Z $')

        ax = plt.subplot(n_rows,n_cols,4)
        utils.plot_scatter([psnr, logprob], ax)

        idx_list = list(range(len(psnr)))
        z_infer, x_recons = self.get_recons_data()
        x_recons = self.denormalize_data(x_recons)
        random.shuffle(idx_list)
        for i in idx_list[:10]:
            im = OffsetImage(x_recons[i])
            ab = AnnotationBbox(im, [psnr[i],logprob[i]],frameon=False)
            ax.add_artist(ab)

        f.suptitle(self.model_name)

        return f









    '''  ------------------------------------------------------------------------------
                                         GENERATOR GRAPH CLASS
        ------------------------------------------------------------------------------ '''


class GeneratorGraph():
    def __init__(self,z_dim, input_G_tensor,  output_G_tensor, placeholder_dict = dict()):

        self.in_G = input_G_tensor
        self.out_G = output_G_tensor
        self.z_dim = z_dim
        self.placeholder_dict = placeholder_dict

    def create_graph(self,meta_file = None):
        with tf.variable_scope('inputs'):
            self.z_input = tf.placeholder(tf.float32,[None, self.z_dim], name='z')

        input_map = self.placeholder_dict.copy()
        input_map[self.in_G] = self.z_input
        new_saver = tf.train.import_meta_graph(meta_file,
                                               input_map=input_map,
                                               import_scope='import')
        self.x_recons = tf.get_default_graph().get_tensor_by_name(self.out_G)

        return new_saver


    def get_recons(self, session, z):
        feed = {self.z_input: z}
        x_recons = session.run(self.x_recons,  feed_dict=feed)
        return x_recons
