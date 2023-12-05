import numpy as np

from abc import ABC, abstractmethod
from enum import Enum


class HiddenActvFuncType(Enum):
    LINEAR = 1
    RELU = 2
    
    
class OneHotEncoder():
    def __init__(self, classes: np.ndarray):
        self.classes = np.unique(classes.flatten())
        self.classes.sort()
        self.n_classes = self.classes.shape[0]
        self.class_map = dict([(self.classes[i], i) for i in range(self.n_classes)])
        
    def print_hot_encoding(self) -> None:
        for label in self.class_map.keys():
            print("{0}: {1}".format(label, self.encode(label)))
            
    def encode(self, label) -> np.ndarray:
        val = self.class_map.get(label, None)
        
        if val is None:
            raise Exception("Categoric class could not be recognized.")
            
        return np.array([1 if val == i else 0 for i in range(self.n_classes)])
    
    def decode(self, code: np.ndarray):
        return self.classes[code.argmax()]
    
    def batch_encode(self, labels: np.ndarray) -> np.ndarray:
        return np.array([self.encode(x) for x in labels])
    
    def batch_decode(self, codes: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.decode, 1, codes)
    

class Loss():
    def __init__(self):
        self.loss_val = None
    
    @abstractmethod
    def measure(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        return None
    
    @abstractmethod
    def gradient(self, target: np.ndarray) -> np.ndarray:
        return None
    
    def batch_measure(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.measure, 1, y_pred, y)
    

class CrossEntropyLoss(Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
            
    def measure(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        self.loss_val = -(y * np.log(y_pred)).sum()
        return self.loss_val
    
    def gradient(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        indx = target.argmax()
        return np.array([(-1 / pred[indx]) if i == indx else 0 for i in range(target.shape[0])])


class HiddenActivationFunction():
    def __init__(self, actv_type: HiddenActvFuncType, opts: dict=None):
        if opts is None:
            opts = {}
            
        self.type = actv_type
            
        if actv_type == HiddenActvFuncType.LINEAR:
            self.activate = lambda x: x
            self.gradient = self.__linear_grad
        elif actv_type == HiddenActvFuncType.RELU:
            self.alpha = opts.get('alpha', 0.0)
            self.activate = np.vectorize(lambda x : max(self.alpha * x, x))
            self.gradient = self.__relu_grad
        else:
            raise Exception("Activation type {0} not supported.".format(actv_type))
            
    def __linear_grad(self, z: np.ndarray) -> np.array:
        return np.ones(z.shape[0])
    
    def __relu_grad(self, z: np.ndarray) -> np.array:
        return np.array([1 if z_i >= self.alpha * z_i else self.alpha for z_i in z])

    
class AbstractNNLayer(ABC):
    def __init__(self, include_bias: bool=True):
        self.has_bias = include_bias
        self.in_size = 0
        self.out_size = 0
        self.activations = None
    
    @abstractmethod
    def forward_pass(self, in_x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Performs the forward pass in the neural network layer.
        This is only an abstract method and should be implemented
        by each child layer.
        
        Arguments:
            in_x (np.ndarray): Input sample as an Numpy ndarray.
        Returns:
            (np.ndarray): Numpy ndarray with predicted values.
        """
        return None
    
    @abstractmethod
    def backwards_pass(self, loss_grad, learn_rate):
        return None
    
    @abstractmethod
    def gradient(self) -> np.array:
        return None
    
    def batch_forward_pass(self, in_x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Performs the forward pass in a batch of datasamples.
        
        Arguments:
            in_x (np.ndarray): Input batch sample as an Numpy ndarray. 
                Each row is a sample.
        Returns:
            (np.ndarray): Numpy ndarray with predicted values.
        """
        return np.apply_along_axis(self.forward_pass, 1, in_x)
    
    
class InputLayer(AbstractNNLayer):    
    def __init__(self, in_size: int):
        super(InputLayer, self).__init__(False)         
        self.in_size = in_size
        self.out_size = in_size
            
    def forward_pass(self, in_x: np.ndarray) -> np.ndarray:
        self.activations = in_x
        return in_x  
    
    def backwards_pass(sef, loss_grad, learn_rate):
        pass
    
    def gradient(self) -> np.array:
        return None

    
class FullyConnectedLayer(AbstractNNLayer):
    def __init__(self, in_layer: AbstractNNLayer, n_neurons: int, 
                 include_bias: bool=True, opts: dict=None):    
        super(FullyConnectedLayer, self).__init__(include_bias) 
        
        if opts is None:
            opts = {}
        
        self.n_neurons = n_neurons
        self.in_layer = in_layer        
        self.out_size = n_neurons   
        self.in_size = in_layer.out_size 
        
        # Initialize weights using a normal distribution.
        self.weights = np.random.normal(
            loc=opts.get('w_mean', 0.0), scale=opts.get('w_std', 0.1), 
            size=(self.n_neurons, self.in_size))
        
        if include_bias:
            self.bias = np.random.normal(
            loc=opts.get('w_mean', 0.0), scale=opts.get('w_std', 0.1), 
            size=(self.n_neurons))
        else:
            self.bias = np.zeros(self.n_neurons)
            
        actv_type = opts.get('func', HiddenActvFuncType.LINEAR)
        self.actv_func = HiddenActivationFunction(actv_type, opts)
        
    def get_x_input(self):
        return self.in_layer.activations
        
    def forward_pass(self, in_x: np.ndarray) -> np.ndarray:
        x = self.in_layer.forward_pass(in_x)
        self.z = self.weights.dot(x) + self.bias
        self.activations = self.actv_func.activate(self.z)
        return self.activations
    
    def gradient(self):
        return self.actv_func.gradient(self.z)
    
    def backwards_pass(self, loss_grad, learn_rate):
        # sigma L = sigma^(L+1) * (grad A / grad Z)
        grad = self.gradient()
        loss_grad = loss_grad * grad
        
        # Adjust bias
        if self.has_bias:
            self.bias = self.bias - (learn_rate * loss_grad)
            
        # Update weights
        update = learn_rate * loss_grad[:, np.newaxis] * self.get_x_input()
        loss_grad = loss_grad.dot(self.weights)
        self.weights = self.weights - update
        self.in_layer.backwards_pass(loss_grad, learn_rate)
    
    
class SoftmaxOutLayer(AbstractNNLayer):
    def __init__(self, in_layer: AbstractNNLayer, classes: np.ndarray, 
                 include_bias: bool=True, opts: dict=None):        
        self.__init__(in_layer, OneHotEncoder(classes), 
            include_bias, opts)
        
    def __init__(self, in_layer: AbstractNNLayer, one_hot: OneHotEncoder, 
                     include_bias: bool=True, opts: dict=None):
        super(SoftmaxOutLayer, self).__init__(include_bias)
        
        if opts is None:
            opts = {}
        
        self.one_hot_enc = one_hot
        self.in_layer = in_layer
        self.in_size = in_layer.out_size
        self.out_size = self.one_hot_enc.n_classes
        
        if include_bias:
            self.bias = np.random.normal(
            loc=opts.get('w_mean', 0.0), scale=opts.get('w_std', 0.1), 
            size=(self.out_size))
        else:
            self.bias = np.zeros(self.out_size)
        
        # Initialize weights using a normal distribution.
        self.weights = np.random.normal(
            loc=opts.get('w_mean', 0.0), scale=opts.get('w_std', 0.1), 
            size=(self.out_size, self.in_size))
        
    def get_x_input(self):
        return self.in_layer.activations
            
    def forward_pass(self, in_x: np.ndarray) -> (np.ndarray, np.ndarray):
        x = self.in_layer.forward_pass(in_x)
        self.z = self.weights.dot(x) + self.bias
        # max_val = self.z.max()
        # print(max_val)
        # exp_x = np.exp(self.z - max_val)
        exp_x = np.exp(self.z)
        sum_exp_x = exp_x.sum()
        self.activations = np.array([(x / sum_exp_x) for x in exp_x])
        return self.activations
    
    def predict(self, in_x: np.ndarray):
        return self.one_hot_enc.decode(self.forward_pass(in_x))
    
    def batch_predict(self, in_x: np.ndarray):
        return np.apply_along_axis(self.predict, 1, in_x)
    
    def gradient(self) -> np.array:
        return np.array([(a * (1 - a)) for a in self.activations]) 
    
    def backwards_pass(self, loss_grad, learn_rate):           
        # sigma L = sigma^(L+1) * (grad A / grad Z)
        # loss_grad = loss_grad * self.gradient()
        loss_grad
        
        # Adjust bias
        if self.has_bias:
            self.bias = self.bias - (learn_rate * loss_grad)
        
        # Adjust weights
        updates = learn_rate * loss_grad[:, np.newaxis] * self.get_x_input()
        loss_grad = loss_grad.dot(self.weights)
        self.weights = self.weights - updates
        self.in_layer.backwards_pass(loss_grad, learn_rate)