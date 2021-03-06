# coding=utf-8
import numpy as np
from sklearn.utils import shuffle

def onehot(x,y):
    m = x.shape[0]
    n = len(np.unique(y))
    b = np.zeros((m,n))
    for i in range(m):
        b[i,y[i]] = 1

    return b
def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    
    return x*(1-x)


def tanh(x):
   
    return np.tanh(x)


def dtanh(x):

        return 1 - (x ** 2)


def softmax(X):
   
    return np.exp(X).T / (np.sum(np.exp(X), axis=1)).T


class MLP:
    def __init__(self, input_size, output_size, hidden_layer_size=[100], batch_size=200, activation="sigmoid",
                 output_layer='softmax', loss='cross_entropy', lr=0.01, reg_lambda=0.0001, momentum=0.9, verbose=10):
      
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_layers = len(hidden_layer_size)  # only hidden layers
        self.activation = activation
        self.verbose = verbose

        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_dfunc = dsigmoid
        elif activation == 'tanh':
            self.activation_func = tanh
            self.activation_dfunc = dtanh
        else:
            raise ValueError("Currently only supoort 'sigmoid' or 'tanh' activations func!")

        self.loss = loss
        if output_layer == 'softmax':
            self.output_layer = softmax
        else:
            raise ValueError('Currently only Support softmax output_layer!')

        self.nclass = output_size

        self.weights = []  # store weights
        self.bias = []  # store bias
        self.layers = []  # store forwarding activation values
        self.deltas = []  # store errors for backprop


    def get_weight_bound(self, fan_in, fan_out):
        if self.activation == 'sigmoid':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation == 'tanh':
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        return init_bound

    def fit(self, X, y, max_epochs, shuffle_data):
            n_samples, n_features = X.shape
            if y.shape[0] != n_samples:
                raise ValueError("Shapes of X and y don't fit!")

            # generate weights
            # Weights and bias connecting input layer and first hidden layer
            init_bound = self.get_weight_bound(n_features, self.hidden_layer_size[0])
            self.weights.append(np.random.uniform(-init_bound, init_bound, size=(n_features, self.hidden_layer_size[0])))

                       self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[0]))

            # Weights and bias connecting hidden layers
            for i in range(1, len(self.hidden_layer_size)):
                init_bound = self.get_weight_bound(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])
                self.weights.append(np.random.uniform(-init_bound, init_bound,
                                                      size=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])))
                
                self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[i]))

            # Weights and bias connecting last hidden layer and output layer
            init_bound = self.get_weight_bound(self.hidden_layer_size[-1], self.output_size)
            self.weights.append(
                np.random.uniform(-init_bound, init_bound, size=(self.hidden_layer_size[-1], self.output_size)))
            self.bias.append(np.random.uniform(-init_bound, init_bound, self.output_size))

            # pre-allocate memory for both activations and errors
            # for input layer
            self.layers.append(np.empty((self.batch_size, self.input_size)))
            # for hidden layers
            for i in range(0, self.n_layers):
                self.layers.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
                self.deltas.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
            # for output layer
            self.layers.append(np.empty((self.batch_size, self.output_size)))
            self.deltas.append(np.empty((self.batch_size, self.output_size)))

            # main loop
            for i in xrange(max_epochs):

                # shuffle data
                
                order = np.asarray(range(0,X.shape[0]))
                np.random.shuffle(order)
                X = X[order,:]
                y = y[order]

                # iterate every batch
                for batch in xrange(0, n_samples, self.batch_size):
                    # TODO call forward function
                    # TODO call backward function
                    self.forward(X[batch:batch + self.batch_size,:])
                    self.backward(X[batch:batch + self.batch_size],y[batch:batch + self.batch_size])

                if i % self.verbose == 0:
                    # Compute Loss and Training Accuracy
                    loss = self.compute_loss(X, y)
                    acc = self.score(X, y)
                    print('Epoch {}: loss = {}, accuracy = {}'.format(i, loss, acc))

            return self

    def compute_loss(self, X, y):
        """
        Compute loss
        :param X: data, array-like, shape(n_sample, n_feature)
        :param y: label, array-like, shape(n_sample, 1)
        :return: loss value
        """
        n_samples = X.shape[0]
        probs = self.forward(X)
        epslon = 1e-40
        
        # calculate the cross-entropy loss
        Y = onehot(X,y)
        data_loss  = -1. * np.sum(Y * np.log(probs + epslon ) + (1 - Y) * np.log(1 - probs + epslon))


        for i in range(0,len(self.hidden_layer_size)):
            data_loss += 0.5 * self.reg_lambda * np.sum(np.power(self.weights[i],2))

        return 1. / n_samples * data_loss

    def forward(self, X):
        # input layer
        self.layers[0] = X

        # hidden layers
        for i in range(1,len(self.hidden_layer_size)+1):
            self.layers[i] = self.activation_func(np.dot(self.layers[i - 1],self.weights[i - 1]) + self.bias[i - 1])

        # output layer (Note here the activation is using output_layer func)
        self.layers[self.n_layers + 1] = self.activation_func(np.dot(self.layers[self.n_layers],self.weights[self.n_layers]) + self.bias[self.n_layers])

        return self.layers[-1]

    def backward(self, X, y):
        n_sample = X.shape[0]
        if self.loss == 'cross_entropy':
            self.deltas[-1] = self.layers[-1]
            # cross_entropy loss backprop
            self.deltas[-1][range(X.shape[0]), y] -= 1

            # update deltas
            for i in range(self.n_layers,0,-1):
                self.deltas[i - 1] = np.dot(self.deltas[i],self.weights[i].T) * self.activation_dfunc(self.layers[i])
            # TODO update weights
            for i in range(self.n_layers,-1,-1):
                # dw = np.dot(self.layers[i].T,self.deltas[i])
                dw = np.dot(self.layers[i].T,self.deltas[i]) / X.shape[0]
                # 这里还没有加regulations terms
                self.weights[i] -= self.lr * dw


    def predict(self, X):
        return self.forward(X)

    def score(self, X, y):
        n_samples = X.shape[0]

        # compute accuracy
        pred = self.forward(X)
        # argamx返回的是最大值的下标
        result_list = pred.argmax(axis = 1) == y

        return np.sum(result_list * 1.0 / n_samples)


def my_mlp():
    # from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    dataset = sklearn.datasets.load_digits()
    # X是1500 * 64
    # 一共有64个features
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    # Y就是1500的列向量
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    network = MLP(input_size=64, output_size=10, hidden_layer_size=[128], batch_size=200, activation="sigmoid",
                  output_layer='softmax', loss='cross_entropy', lr=0.1)

    network.fit(X_train, y_train, 100, True)

    acc = network.score(X_test, y_test)
    print('Test Accuracy: {}'.format(acc))


def sklearn_mlp():
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_mldata
    from sklearn.neural_network import MLPClassifier

    # mnist =fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    dataset = sklearn.datasets.load_digits()
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=100, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.01)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))


def main():
    print('Class 2 Multiple Layer Perceptron (MLP) Example')
    my_mlp()

    print ('')

    print('Class 2 sklearn MLP Example')
    sklearn_mlp()


if __name__ == "__main__":
    main()
