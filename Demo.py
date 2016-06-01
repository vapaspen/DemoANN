__author__ = 'vapaspen'
__name__ = 'Demo'
'''This is a Demo ANN that looks at a given set of context data and check if analysis is realated to it.
A match should be if a given analysis entry has the same 1 in the right most position.

Example:
context:
    [1, 1, 0]
Matches:
Analysis:
    [1, 1, 0]
    [0, 1, 0]

'''

import numpy as np
from scipy import stats
import random
from ANN import ANN
from RNN import RNN

#Demo Data to train the Network
data = {
    '0':{
        'context': [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        'analysis': [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        'target': [
            [1, 0],
            [1, 0],
            [0, 1]
        ]
    },
    '1': {
        'context': [
            [1, 1, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        'analysis': [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ],
        'target': [
            [1, 0],
            [0, 1],
            [0, 1]
        ]
    },
    '2': {
        'context': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ],
        'analysis': [
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 1]
        ],
        'target': [
            [0, 1],
            [1, 0],
            [0, 1]
        ]
    },
    '3': {
        'context': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ],
        'analysis': [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0]
        ],
        'target': [
            [1, 0],
            [0, 1],
            [1, 0]
        ]
    },
    '4': {
        'context': [
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1]
        ],
        'analysis': [
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 1]
        ],
        'target': [
            [1, 0],
            [0, 1],
            [1, 0]
        ]
    },
    '5': {
        'context': [
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1]
        ],
        'analysis': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1]
        ],
        'target': [
            [0, 1],
            [0, 1],
            [1, 0]
        ]
    }

}





print("*****Building Model*****")
#Create all of the layers each with a unique starting random state.

rng = np.random.RandomState(np.random.randint(low=10, high=10000))
FeatureLayerIn = ANN(Xc=3 , Hc=6, rng=rng)
rng = np.random.RandomState(np.random.randint(low=10, high=10000))
FeatureLayerOut = ANN(Xc=6 , Hc=3, rng=rng)


rng = np.random.RandomState(np.random.randint(low=10, high=10000))
ContextIn = RNN(Xc=3 , Hc=6, rng=rng)
rng = np.random.RandomState(np.random.randint(low=10, high=10000))
ContextOut = RNN(Xc=6 , Hc=2, rng=rng)

rng = np.random.RandomState(np.random.randint(low=10, high=10000))
AnalysisLayer = ANN(Xc=5 , Hc=2, rng=rng)


epoch = 0 #Iteration counter

display = 1000 #Number of iterations between displays

epoch_list = list(range(len(data))) #list of used for training order
print('*****Training Model*****')
print()
while epoch < 1000000:
    random.shuffle(epoch_list)

    for i in epoch_list:

        #reset the Layer stats at the start of each data block in the epoch
        FeatureLayerIn.reset()
        FeatureLayerOut.reset()
        ContextIn.reset()
        ContextOut.reset()
        AnalysisLayer.reset()
        sample = []

        #Process the context data in this block
        for item in data[str(i)]['context']:
            #Uses a Feedfoward process that down samples by half as the feature detectors.
            #The first Layer expands the dimensionality to be able to represent more possible types of inputs.
            FLI = FeatureLayerIn.FF(item)
            FLI = stats.threshold(FLI, threshmin=np.median(FLI), newval=0)
            FLO = FeatureLayerOut.FF(FLI)
            FL0 = stats.threshold(FLO, threshmin=np.median(FLO), newval=0)

            #recurrent Layers
            CLI = ContextIn.FF(FLO)
            CLO = ContextOut.FF(CLI)

        for item in data[str(i)]['analysis']:
            #same as the Context layer
            FLI = FeatureLayerIn.FF(item)
            FLI = stats.threshold(FLI, threshmin=np.median(FLI), newval=0)
            FLO = FeatureLayerOut.FF(FLI)
            FLO = stats.threshold(FLO, threshmin=np.median(FLO), newval=0)

            #Final analysis layer that make a determination based on the results of the context layer plus the
            #feture mapped input.
            analysisData = np.concatenate((FLO, CLO), axis=0)

            #Save the results for this time step
            sample.append(AnalysisLayer.FF(analysisData))

        #Process for updating parameters by computing stochastic gradient descent
        for t in reversed(range(len(sample))):
            error = sample[t] - np.array(data[str(i)]['target'][t]) #Dervitive of 1/2 Mean squared Error

            dAL = AnalysisLayer.Grad(error, t)

            #Compute the gradient of the FeatureLayers with respect to the Analysis input
            dFLOa = FeatureLayerOut.Grad(dAL[:3], t)
            FeatureLayerIn.Grad(dFLOa, t)

            #Unreavale the context layer with respect to the analysis input to compute the gradient.
            for ts in reversed(range(len(data[str(i)]['context']))):
                dCLO = ContextOut.Grad(dAL[3:], ts)
                dCLI = ContextIn.Grad(dCLO, ts)

                #Get new update reacted to Time Sub with respect to both the context input and the Analysis result
                #Add these gradients to the existing calculated values generated at the Analysis stage.
                dFLOc = FeatureLayerOut.Grad(dCLI, ts)
                FeatureLayerIn.Grad(dFLOc, ts)

            #if we are on a Display epoch display the data give for each analysis, the result, and the target
            # for what it should have been
            if epoch % display == 0:
                print(str(data[str(i)]['analysis'][t]) + " " + str(sample[t]) + " " + str(data[str(i)]['target'][t]))
        #update the layers for this Data Block
        FeatureLayerIn.update(learning_rate=0.1)
        FeatureLayerOut.update(learning_rate=.1)
        ContextIn.update(learning_rate=.1)
        ContextOut.update(learning_rate=.1)
        AnalysisLayer.update(learning_rate=.1)

        #add an epoch number to help with display
        if epoch % display == 0:
            print("Epoch: " + str(epoch))
            print()
    epoch += 1

