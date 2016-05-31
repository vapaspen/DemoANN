__author__ = 'vapaspen'
__name__ = 'ANN'

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ANN (object):
    """
        RNN Layer using only Numpy.

        FF: Feed Foward method
        Grad: Method to get Gradients
    """

    def __init__(self, Xc=None, Hc=None, rng=None):

        """RNN class to make a simple connected RNN layer

        :param Xc: Total input count
        :param Hc: Total Node count
        :param rng: numpy.random.RandomState
        """

        self.params = {}

        self.deltas = []

        self.params["b"] = np.zeros((Hc,), dtype=float)
        self.params["Wx"] = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (Xc + Hc)),
                    high=np.sqrt(6. / (Xc + Hc)),
                    size=(Xc, Hc)
                ),
                dtype=float
        )*4

        self.delta = {}
        self.hist = []
        self.reset()


    def reset(self):
        """ Resets all persistence variables

        :return: Void
        """
        self.hist = [
            {
                'Y': np.zeros_like(self.params['b']),
                'X': np.zeros((len(self.params['Wx']),), dtype=float)
            }
        ]

        self.deltas = {
            'db': np.zeros_like(self.params['b']),
            'dWx': np.zeros_like(self.params['Wx']),
        }


    def FF(self, input, reset=False):
        """ Method for Feeding Fowared on the Layer.

        :param input: Array input
        :param reset: Bool for if the history needs to be reset

        :return: Returns the result node vaules.
        """

        if reset:
            self.reset()

        if not len(input) == len(self.params["Wx"]):
            raise Exception("Input given not the same shape as Layer settings.")

        x = input
        #y = np.tanh(np.dot(x, self.params["Wx"]) + np.dot(yLast, self.params["Wh"]) + self.params["b"])
        y = sigmoid(np.dot(x, self.params["Wx"])  + self.params["b"])

        self.hist.append({
            'X':x,
            'Y':y
        })

        return y

    def Grad(self, error, t):
        """

        :param error: the Gradient Error at the post activation of this layer.
        :param t: Current Time to operate on.
        :return: The Gradient Error for the Input of this Layer.
        """
        t +=1

        dY = error

        #db = ((1 - self.hist[t]['Y']) * self.hist[t]['Y']) * dY
        db = (self.hist[t]['Y'] * (1 - self.hist[t]['Y'])) * dY


        dWx = db * np.array([self.hist[t]['X']]).T

        dX = np.dot(db, self.params["Wx"].T)

        self.deltas['db'] += db
        self.deltas['dWx'] += dWx


        return dX


    def update(self, learning_rate= 0.01):

        self.params['b'] += self.deltas['db'] * -learning_rate
        self.params['Wx'] += self.deltas['dWx'] * -learning_rate


    def save(self, location):

        np.save(location, self.params)

    def load(self, loaction):
        self.params = np.load(loaction).item()