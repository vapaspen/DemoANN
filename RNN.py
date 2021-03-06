__author__ = 'vapaspen'
__name__ = 'RNN'

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RNN (object):
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
        self.params["Wh"] = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (Hc + Hc)),
                high=np.sqrt(6. / (Hc + Hc)),
                size=(Hc, Hc)
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
            'dWh': np.zeros_like(self.params['Wh']),
            'dWx': np.zeros_like(self.params['Wx']),
            'dY_h': np.zeros_like(self.params['b']),
        }


    def FF(self, input, reset=False):
        """ Method for Feeding Forward on the Layer.

        :param input: Array input
        :param reset: Bool for if the history needs to be reset

        :return: Returns the result node vales.
        """

        if reset:
            self.reset()

        if not len(input) == len(self.params["Wx"]):
            raise Exception("Input given not the same shape as Layer settings.")
        yLast = self.hist[len(self.hist)-1]['Y']
        x = input
        y = sigmoid(np.dot(x, self.params["Wx"]) + np.dot(yLast, self.params["Wh"]) + self.params["b"])

        self.hist.append({
            'X':x,
            'Y':y
        })

        return y

    def Grad(self, error, t):
        """ Function to compute the gradients at time T

        :param error: the Gradient Error at the post activation of this layer.
        :param t: Current Time to operate on.
        :return: The Gradient Error for the Input of this Layer.
        """
        t +=1  #increments Time because the gradients need to start at time -1

        dY = error + self.deltas['dY_h']

        db = (self.hist[t]['Y'] * (1 - self.hist[t]['Y'])) * dY


        dWx = db * np.array([self.hist[t]['X']]).T
        dWh = db * np.array([self.hist[t - 1]['Y']]).T

        dY_h = np.dot(db, self.params["Wh"].T)
        dX = np.dot(db, self.params["Wx"].T)

        self.deltas['db'] += db
        self.deltas['dWx'] += dWx
        self.deltas['dWh'] += dWh
        self.deltas['dY_h'] += dY_h


        return dX


    def update(self, learning_rate= 0.01):
        """Function to update the Parameters based on the computed gradients.

        :param learning_rate: the Learning rate for the update

        :return: void
        """

        self.params['b'] += self.deltas['db'] * -learning_rate
        self.params['Wx'] += self.deltas['dWx'] * -learning_rate
        self.params['Wh'] += self.deltas['dWh'] * -learning_rate


    def save(self, location):
        """ Function to save the parameter state to a file

        :param location: Location to save to
        :return: void
        """
        np.save(location, self.params)

    def load(self, location):
        """ Function to Load the layer parameter state from a file

        :param location:  Location to Load
        :return: Void
        """

        self.params = np.load(location).item()