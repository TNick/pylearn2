"""
Retreived from 
https://github.com/shamidreza/pylearn2_wrapper/blob/master/pylearn2_wrapper.py

GNU GENERAL PUBLIC LICENSE Version 2, June 1991

pylearn2 Wrapper script
Author: Seyed Hamidreza Mohammadi
Date: 05/13/2014
Contains:
  RegularAutoencoder  
  DenoisingAutoencoder
  ContractiveAutoencoer
  HigherOrderContractiveAutoencoer
  DeepGeneralAutoencoder
  MLP (Multilayer Perceptron)
"""

from abc import ABCMeta, abstractmethod
import copy
import pickle
import logging 

# third party library
import numpy as np
import theano
import pylearn2
import pylearn2.train        
import pylearn2.models.mlp
import pylearn2.training_algorithms.sgd
import pylearn2.costs.mlp.dropout
import pylearn2.termination_criteria
import pylearn2.models.autoencoder
import pylearn2.costs.autoencoder
import pylearn2.train        
import pylearn2.models.mlp
import pylearn2.training_algorithms.sgd
import pylearn2.costs.mlp.dropout
import pylearn2.termination_criteria

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Autoencoder(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, X):
        pass
    @abstractmethod
    def encode(self, X):
        pass 
    @abstractmethod
    def decode(self, X):
        pass    
    @abstractmethod
    def reconstruct(self, X):
        pass    
    def save(self, filename):
        f = open(filename, 'w+')
        pickle.dump(self, f)
        f.flush()
        f.close()
    @classmethod
    def load(cls, filename):
        f = open(filename, 'r')
        ae = pickle.load(f)
        f.close()
        return ae  
        
            
class DeepGeneralAutoencoder(Autoencoder):
    """ This class should be able to stack any type of AEs,
    It should be a wrapper to train them layer-by-layer since pylearn2 only have ability to learn the layers
    all at once only for Regular AEs """
    def __init__(self, autoencoders):
        for i in range(len(autoencoders)):
            assert isinstance(autoencoders[i], RegularAutoencoder), 'Not an Autoencoder.'
        self.model = autoencoders
        self.config = {}
    def train(self, X, num_iterations=1000, learning_rate=0.01, batch_size=50):
        self.config.update( {'iterations':num_iterations, 'learning_rate':learning_rate, 'batch_size':batch_size})        
        autoencoders = self.model
        tmp = copy.deepcopy(X)
        for i in range(len(autoencoders)):
            autoencoders[i].train(tmp, num_iterations, learning_rate, batch_size)
            tmp = autoencoders[i].encode(tmp)
    def encode(self, X):
        autoencoders = self.model        
        tmp = copy.deepcopy(X)        
        for i in range(len(autoencoders)):
            tmp = autoencoders[i].encode(tmp)
        return tmp
    def decode(self, X):  
        autoencoders = self.model        
        tmp = copy.deepcopy(X)        
        for i in range(len(autoencoders)-1, -1, -1):
            tmp = autoencoders[i].decode(tmp)
        return tmp  
    def reconstruct(self, X):
        return self.decode(self.encode(X))
     
class DeepRegularAutoencoder(Autoencoder):
    """ This class only stacks Regular AEs """
    def __init__(self, layers, act_enc_types, act_dec_types, input_dimension):
        assert len(layers) == len(act_enc_types), 'the layers info do not match.'
        assert len(act_dec_types) == len(act_enc_types), 'the layers info do not match.'
        
        if not hasattr(self, 'config'):
            self.config = {}
        for i in range(len(act_dec_types)):
            if act_dec_types[i] == 'linear':
                act_dec_types[i] = None
        for i in range(len(act_enc_types)):
            if act_enc_types[i] == 'linear':
                act_enc_types[i] = None                
        self.config.update({'layers':layers, 'act_enc_types':act_enc_types, 'act_dec_types':act_dec_types, 'input_dimension':input_dimension})
        self._build_model()
    def _build_model(self):
        layers = self.config['layers']
        aelayers = []
        aelayers.append(pylearn2.models.autoencoder.Autoencoder(self.config['input_dimension'], layers[0], self.config['act_enc_types'][0], self.config['act_dec_types'][0], irange=0.05))
        for i in range(1,len(layers)):
            aelayers.append(pylearn2.models.autoencoder.Autoencoder(layers[i-1], layers[i], self.config['act_enc_types'][i], self.config['act_dec_types'][i], irange=0.1))
    
        self.model = pylearn2.models.autoencoder.DeepComposedAutoencoder(
            autoencoders=aelayers               
        )        
    def train(self, X, num_iterations=1000, learning_rate=0.01, batch_size=50):
        self.config.update( {'iterations':num_iterations, 'learning_rate':learning_rate, 'batch_size':batch_size})
        
        algorithm = pylearn2.training_algorithms.sgd.SGD(
                    learning_rate = self.config['learning_rate'],
                    batch_size = batch_size,
                    batches_per_iter=int(X.shape[0]/batch_size),
                    cost = pylearn2.costs.autoencoder.MeanSquaredReconstructionError(),            
                    termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations']),
                )
        
        
        
        data_train = pylearn2.datasets.DenseDesignMatrix(X=X[:,:])
        
        train = pylearn2.train.Train(
                    dataset = data_train,
                    model = self.model,
                    algorithm = algorithm,
                    save_path = 'tmp.pkl',
                    save_freq = 100,
                    extensions=[pylearn2.training_algorithms.sgd.OneOverEpoch(self.config['iterations']//2, self.config['iterations']//8)]
        )
        
        train.main_loop()
    def encode(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.encode(X_theano).eval()
    def decode(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.decode(X_theano).eval()
    def reconstruct(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.decode(self.model.encode(X_theano)).eval()
   
class RegularAutoencoder(Autoencoder):
    def __init__(self, layer, act_enc_type, act_dec_type, input_dimension, tied=True):        
        if not hasattr(self, 'config'):
            self.config = {}
        if act_dec_type == 'linear':
            act_dec_type = None
        if act_enc_type == 'linear':
            act_enc_type = None                
        self.config.update({'tied':tied,'layer':layer, 'act_enc_type':act_enc_type, 'act_dec_type':act_dec_type, 'input_dimension':input_dimension})
        self._build_model()
        self.algorithm = None
    def _build_model(self):        
        self.model = pylearn2.models.autoencoder.Autoencoder(self.config['input_dimension'],
                                                             self.config['layer'],
                                                             self.config['act_enc_type'],
                                                             self.config['act_dec_type'],
                                                             tied_weights=self.config['tied'],
                                                             irange=0.05)
                
    def _build_algorithm(self, num_samples):
        self.algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            batch_size = self.config['batch_size'],
            batches_per_iter=int(num_samples/self.config['batch_size']),
            cost = pylearn2.costs.autoencoder.MeanSquaredReconstructionError(),            
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations']),
        )
        
    def train(self, X, num_iterations=1000, learning_rate=0.01, batch_size=50):
        self.config.update( {'iterations':num_iterations, 'learning_rate':learning_rate, 'batch_size':batch_size})        
        data_train = pylearn2.datasets.DenseDesignMatrix(X=X[:,:])
        self._build_algorithm(X.shape[0])
        train = pylearn2.train.Train(
                    dataset = data_train,
                    model = self.model,
                    algorithm = self.algorithm,
                    save_path = 'tmp.pkl',
                    save_freq = 100,
                    extensions=[pylearn2.training_algorithms.sgd.OneOverEpoch(max(1,self.config['iterations']//2), max(1,self.config['iterations']//8))]
        )
        
        train.main_loop()
    def encode(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.encode(X_theano).eval()
    def decode(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.decode(X_theano).eval()
    def reconstruct(self, X):
        X_theano = theano.shared(value=X, name='inputs')
        return self.model.decode(self.model.encode(X_theano)).eval()
    
class SparseL1Autoencoder(RegularAutoencoder):  
    def _build_model(self):         
        self.model = pylearn2.models.autoencoder.Autoencoder(self.config['input_dimension'],
                                                                        self.config['layer'],
                                                                        self.config['act_enc_type'],
                                                                        self.config['act_dec_type'],
                                                                        tied_weights=self.config['tied'],
                                                                        irange=0.05)
    def _build_algorithm(self, num_samples):
        self.algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            batch_size = self.config['batch_size'],
            batches_per_iter=int(num_samples/self.config['batch_size']),
            #cost = pylearn2.costs.cost.SumOfCosts([pylearn2.costs.autoencoder.MeanBinaryCrossEntropy(),
                                                  #[0.1, pylearn2.costs.cost.MethodCost('contraction_penalty')]]),
            cost = pylearn2.costs.cost.SumOfCosts(
                [[1.0,pylearn2.costs.autoencoder.MeanSquaredReconstructionError()],
                [0.2, pylearn2.costs.autoencoder.SampledMeanSquaredReconstructionError(0.2,0.00001)]]),
            #cost = pylearn2.costs.autoencoder.SampledMeanSquaredReconstructionError(0.1,1),
            
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations'])
        )  
class SparseKLAutoencoder(RegularAutoencoder):  
    def _build_model(self):         
        self.model = pylearn2.models.autoencoder.Autoencoder(self.config['input_dimension'],
                                                                        self.config['layer'],
                                                                        self.config['act_enc_type'],
                                                                        self.config['act_dec_type'],
                                                                        tied_weights=self.config['tied'],
                                                                        irange=0.05)
    def _build_algorithm(self, num_samples):
        self.algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            batch_size = self.config['batch_size'],
            batches_per_iter=int(num_samples/self.config['batch_size']),
            #cost = pylearn2.costs.cost.SumOfCosts([pylearn2.costs.autoencoder.MeanBinaryCrossEntropy(),
                                                  #[0.1, pylearn2.costs.cost.MethodCost('contraction_penalty')]]),
            cost = pylearn2.costs.cost.SumOfCosts(
                [[1.0,pylearn2.costs.autoencoder.MeanSquaredReconstructionError()],
                [0.2, pylearn2.costs.autoencoder.SparseActivation(1.0, 0.2)]]),
            #cost = pylearn2.costs.autoencoder.SparseActivation(1.0, 0.2),
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations'])
        )  

class ContractiveAutoencoder(RegularAutoencoder):  
    def _build_model(self):         
        self.model = pylearn2.models.autoencoder.ContractiveAutoencoder(self.config['input_dimension'],
                                                                        self.config['layer'],
                                                                        self.config['act_enc_type'],
                                                                        self.config['act_dec_type'],
                                                                        tied_weights=self.config['tied'],
                                                                        irange=0.05)
    def _build_algorithm(self, num_samples):
        self.algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            batch_size = self.config['batch_size'],
            batches_per_iter=int(num_samples/self.config['batch_size']),
            #cost = pylearn2.costs.cost.SumOfCosts([pylearn2.costs.autoencoder.MeanBinaryCrossEntropy(),
                                                  #[0.1, pylearn2.costs.cost.MethodCost('contraction_penalty')]]),
            cost = pylearn2.costs.cost.SumOfCosts(
                [[1.0,pylearn2.costs.autoencoder.MeanSquaredReconstructionError()],
                [0.1, pylearn2.costs.cost.MethodCost('contraction_penalty')]]),
            #cost = pylearn2.costs.cost.MethodCost('contraction_penalty'),
            
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations'])
        )  
class HigherOrderContractiveAutoencoder(RegularAutoencoder):
    def __init__(self, layer, act_enc_type, act_dec_type, input_dimension, corruptor, tied=True):
        self.config={'corruption':corruptor}
        RegularAutoencoder.__init__(self, layer, act_enc_type, act_dec_type, input_dimension, tied)        
            
    def _build_model(self):         
        self.model = pylearn2.models.autoencoder.HigherOrderContractiveAutoencoder(self.config['corruption'].corruptor,
                                                                                   1,
                                                                                   self.config['input_dimension'],
                                                                                   self.config['layer'],
                                                                                   self.config['act_enc_type'],
                                                                                   self.config['act_dec_type'],
                                                                                   tied_weights=self.config['tied'],
                                                                                   irange=0.05)
    def _build_algorithm(self, num_samples):
        self.algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            batch_size = self.config['batch_size'],
            batches_per_iter=int(num_samples/self.config['batch_size']),
            cost = 
            pylearn2.costs.cost.SumOfCosts(
                [[1.0,pylearn2.costs.autoencoder.MeanSquaredReconstructionError()],
                 [0.1, pylearn2.costs.cost.MethodCost('contraction_penalty')],
                [0.1, pylearn2.costs.cost.MethodCost('higher_order_penalty')]
                ]),                    
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations'])
        )     
class Corruptor:
    def __init__(self, corruptor_type, corruption_level, random_seed=0):
        """corruptor_type: (comments from pylearn2)
            'Gaussian',  A Gaussian corruptor transforms inputs by adding zero mean isotropic Gaussian noise.
            'Binomial',  A binomial corruptor that sets inputs to 0 with probability 0 < `corruption_level` < 1.
            'Dropout', Sets inputs to 0 with probability of corruption_level and then divides by 1 - corruption_level to keep expected activation constant.
	    """
	self.corruptor = None
	if corruptor_type == 'Gaussian':
	    self.corruptor = pylearn2.corruption.GaussianCorruptor(corruption_level, random_seed)
	elif corruptor_type == 'Binomial':
	    self.corruptor = pylearn2.corruption.BinomialCorruptor(corruption_level, random_seed)
	elif corruptor_type == 'Dropout':
	    self.corruptor = pylearn2.corruption.DropoutCorruptor(corruption_level, random_seed)
	else:
	    raise NotImplementedError, 'only Gaussian, Binomial, and Dropout corruptors are covered'
    
class DenoisingAutoencoder(RegularAutoencoder):
    def __init__(self, layer, act_enc_type, act_dec_type, input_dimension, corruptor, tied=True):
        self.config={'corruption':corruptor}
	RegularAutoencoder.__init__(self, layer, act_enc_type, act_dec_type, input_dimension, tied)	
    
    def _build_model(self): 
	self.model = pylearn2.models.autoencoder.DenoisingAutoencoder(self.config['corruption'].corruptor,
	                                                              self.config['input_dimension'],
	                                                              self.config['layer'],
	                                                              self.config['act_enc_type'],
	                                                              self.config['act_dec_type'],
	                                                              tied_weights=self.config['tied'],
	                                                              irange=0.05)
	
class MLP():
    def __init__(self, layers, layers_type, input_dimension):
        # layers_type can be 'tanh' or 'sigmoid' or 'linear' e.g. layers_type=['tanh', 'tanh', 'linear']
        # the last element on layers and layers_type belongs to the output layer
        # so layers[-1] should be equal to Y.shape[1] in train
        assert len(layers) == len(layers_type), 'Layers info do not match.'
        self.config = {'layers':layers, 'layers_type':layers_type, 'input_dimension':input_dimension}
        l = self.config['layers']
        layers = []
        for i in range(len(l)):
            if self.config['layers_type'][i] == 'tanh':
                layers.append(pylearn2.models.mlp.Tanh(layer_name='h'+str(i), dim=l[i], irange=.05))
            elif self.config['layers_type'][i] == 'linear':
                layers.append(pylearn2.models.mlp.Linear(layer_name='h'+str(i), dim=l[i], irange=.05))
            elif self.config['layers_type'][i] == 'sigmoid':
                layers.append(pylearn2.models.mlp.Sigmoid(layer_name='h'+str(i), dim=l[i], irange=.05))                
            else: 
                raise Exception(self.config['layers_type'][i]+' is not a legal layer.')
        #layers.append(pylearn2.models.mlp.Linear(layer_name='h'+str(len(l)), dim=self.config['layers'][-1], irange=.05))
        self.model = pylearn2.models.mlp.MLP(
            layers=layers,
            input_space=None,
            nvis=self.config['input_dimension']
        )
    def train(self, X, Y, num_iterations=1000, learning_rate=0.01, batch_size=50):
        assert Y.shape[1] == self.config['layers'][-1], 'output dimensions do not match.'
        assert X.shape[1] == self.model.layers[0].get_weights().shape[0], 'input dimensions do not match.'
        
        self.config.update({'iterations':num_iterations, 'learning_rate':learning_rate, 'batch_size':batch_size})
        
        algorithm = pylearn2.training_algorithms.sgd.SGD(
            learning_rate = self.config['learning_rate'],
            #init_momentum = 0.5,
            batch_size=self.config['batch_size'],
            batches_per_iter=X.shape[0]//self.config['batch_size'],
            cost = pylearn2.costs.mlp.Default(),            
            termination_criterion = pylearn2.termination_criteria.EpochCounter(self.config['iterations'])
        )
        
        
        data_train = pylearn2.datasets.DenseDesignMatrix(X=X, y=Y)

        train = pylearn2.train.Train(
            dataset = data_train,
            model = self.model,
            algorithm = algorithm,
            save_path = 'tmp.pkl',
            save_freq = 100)
        
        train.main_loop()
        
    def test(self, X):
        assert X.shape[1] == self.model.layers[0].get_weights().shape[0], 'input dimensions do not match.'        
        X_theano = theano.shared(value=X, name='inputs')
        XH = self.model.fprop(X_theano).eval()
        return XH
    
    def initialize_weight_using_autoencoder(self, layer_num, ae, act_layer=True):
        # act_layer specifies wether you want to set the weights from
        # the activation layer of AE (W and hidden bias) or deactivation layer (Wprime and visible bias)
        # note that the activation functions of AE and MLP shoould match
        # e.g. the decoding parts should be linear and encoding parts should be 'tanh'
        try:        
            if act_layer:
                    self.model.layers[layer_num].set_weights(copy.deepcopy(ae.model.weights).eval())
                    self.model.layers[layer_num].set_biases(copy.deepcopy(ae.model.hidbias).eval())
                
            else:
                self.model.layers[layer_num].set_weights(copy.deepcopy(ae.model.w_prime).eval())
                self.model.layers[layer_num].set_biases(copy.deepcopy(ae.model.visbias).eval())        
        except:
            raise Exception('Could not set the weights. check to see if the dimensions match')            
    def initialize_weight_using_nnlayer(self, layer_num, layer):
        # act_layer specifies wether you want to set the weights from
        # the activation layer of AE (W and hidden bias) or deactivation layer (Wprime and visible bias)
        # note that the activation functions of AE and MLP shoould match
        # e.g. the decoding parts should be linear and encoding parts should be 'tanh'
        try:
            self.model.layers[layer_num].set_weights(copy.deepcopy(layer.get_weights()))
            self.model.layers[layer_num].set_biases(copy.deepcopy(layer.get_biases()))                
        except:
            raise Exception('Could not set the weights. check to see if the dimensions match')            
    
    def save(self, filename):
        f = open(filename, 'w+')
        pickle.dump(self, f)
        f.flush()
        f.close()
    @classmethod
    def load(cls, filename):        
        f = open(filename, 'r')
        mlp = pickle.load(f)
        f.close()
        return mlp
    


def script_test_mlp():
    np.random.seed(0)
    if 1: # train data (2-dimensional from a two-variate Gaussian)
        X1 = np.random.multivariate_normal([1,1],[[0.2,0],[0,.2]], 1000)
        X2 = np.random.multivariate_normal([2,2],[[0.2,0],[0,.2]], 1000)    
        X = np.r_[X1, X2]    
        Y1 = np.zeros((1000,1))
        Y2 = np.ones((1000,1))
        Y = np.r_[Y1, Y2] 
    if 1: # test data (class data 1-dimensional 0 or 1)
        X1t = np.random.multivariate_normal([1,1],[[0.2,0],[0,.2]], 1000)
        X2t = np.random.multivariate_normal([2,2],[[0.2,0],[0,.2]], 1000)    
        Xt = np.r_[X1t, X2t]    
        Y1t = np.zeros((1000,1))
        Y2t = np.ones((1000,1))
        Yt = np.r_[Y1t, Y2t]         
    
    mlp = MLP([10, 2, Y.shape[1]], ['tanh', 'sigmoid', 'linear'], X.shape[1])
    mlp.train(X, Y, num_iterations=100)
    mlp.save('savetest.pkl')
    mlp = Autoencoder.load('savetest.pkl')    
    Yh = mlp.test(X)
    Yht = mlp.test(Xt)
    
    print 'training error:', np.mean(np.sqrt(np.mean((Y - Yh)**2,1)))    
    print 'testing error:', np.mean(np.sqrt(np.mean((Yt - Yht)**2,1)))
    
    from matplotlib import pyplot as pp   
    pp.plot(Xt[Yt[:,0]==0,0],Xt[Yt[:,0]==0,1],'b*')
    pp.plot(Xt[Yt[:,0]==1,0],Xt[Yt[:,0]==1,1],'g*')
    pp.show()    
    
    pp.plot(Yt)
    pp.plot(Yht)
    pp.show()
    
def script_test_ae():
    np.random.seed(0)
    X1 = np.random.multivariate_normal([1,1],[[0.1,0],[0,.1]], 1000)
    X2 = np.random.multivariate_normal([2,2],[[0.1,0],[0,.1]], 1000)
    X = np.r_[X1, X2] 
    X1t = np.random.multivariate_normal([1,1],[[0.1,0],[0,.1]], 1000)
    X2t = np.random.multivariate_normal([2,2],[[0.1,0],[0,.1]], 1000)
    Xt = np.r_[X1t, X2t] 
    if 0: # regular AE
        ae = RegularAutoencoder(100, 'tanh', 'linear', X.shape[1],tied=True)
        #training error: 0.011292262243
        #testing error: 0.011583130842
        #training error: 0.00817602383537 RAE_2
        #testing error: 0.00847149293172
        #training error: 0.00637784829721
        #testing error: 0.00643461805906        
    elif 1: # sparse KL AE
        ae = SparseKLAutoencoder(100, 'sigmoid', 'linear', X.shape[1],tied=True)
    elif 1: # sparse L1 AE
        ae = SparseL1Autoencoder(100, 'sigmoid', 'linear', X.shape[1],tied=True) 
        
    elif 0: # denoising AE
        ae = DenoisingAutoencoder(2, 'tanh', 'linear', X.shape[1], Corruptor('Gaussian', 0.02))
        #training error: 0.0156150852744
        #testing error: 0.0162391503726        
    elif 0: # contractive AE
        ae = ContractiveAutoencoder(10, 'tanh', 'linear', X.shape[1])
        #training error: 0.0163085927272
        #testing error: 0.0166757802107
                
    elif 0: # higher order contractive AE
        ae = HigherOrderContractiveAutoencoder(10, 'tanh', 'linear', X.shape[1], Corruptor('Gaussian', 0.02))
        #training error: 0.0163085965283
        #testing error: 0.0166757850085        
    elif 0: # deep regular AE
        ae = DeepRegularAutoencoder([20, 10],['tanh', 'tanh'], ['tanh', 'linear'], X.shape[1])
        #training error: 0.00429356922932 deepRAE_10
        #testing error: 0.00433770115632    
        #training error: 0.0152614391591 deepRAE_10_2
        #testing error: 0.0154029336815 
    elif 0: # deep general AE
        aes = []
        ae = DenoisingAutoencoder(20, 'tanh', 'linear', X.shape[1], Corruptor('Gaussian', 0.02))
        aes.append(ae)
        ae = ContractiveAutoencoder(10, 'tanh', 'tanh', 20)
        aes.append(ae)        
        ae = DeepGeneralAutoencoder(aes)
        #training error: 0.0295310213273
        #testing error: 0.0301444067064
        
        #training error: 0.0195438037217 middle tied dec tanh
        #testing error: 0.0200434872559
        
        #training error: 0.0150618848768 not tied dec tanh
        #testing error: 0.0154372800253        
        
                
    ae.train(X, num_iterations=100, batch_size=5)
    ae.save('savetest.pkl')
    ae = RegularAutoencoder.load('savetest.pkl')
    Xht = ae.reconstruct(Xt)
    Xh = ae.reconstruct(X)
    print 'training error:', np.mean(np.sqrt(np.mean((X - Xh)**2,1)))    
    print 'testing error:', np.mean(np.sqrt(np.mean((Xt - Xht)**2,1)))
    from matplotlib import pyplot as pp   
    pp.plot(Xt[:,0],Xt[:,1],'b*')
    pp.plot(Xh[:,0],Xh[:,1],'r*')
    pp.show()
if __name__ == "__main__":
    script_test_ae()
    #script_test_mlp()