__author__ = 'vapaspen'
__name__ = 'Demo'

import numpy as np
from scipy import stats
import random
from ANN import ANN
from RNN import RNN
'''
data = {
    'context':[
        [1,1,0],
        [0,1,0],
        [0,1,0]
    ],
    'analysis':[
        [1,1,0],
        [0,1,0],
        [0,0,1]
    ],
    'target':[
        [1,0],
        [1,0],
        [0,1]
    ]
}
'''
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
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0]
        ],
        'target': [
            [0, 1],
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
            [1 , 0, 0],
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

epoch = 0
display = 1000
epoch_list = list(range(len(data)))
print('*****Training Model*****')
print()
while epoch < 1000000:
    random.shuffle(epoch_list)

    for i in epoch_list:

        FeatureLayerIn.reset()
        FeatureLayerOut.reset()
        ContextIn.reset()
        ContextOut.reset()
        AnalysisLayer.reset()
        sample = []

        for item in data[str(i)]['context']:
            FLI = FeatureLayerIn.FF(item)
            FLI = stats.threshold(FLI, threshmin=np.median(FLI), newval=0)
            FLO = FeatureLayerOut.FF(FLI)
            FL0 = stats.threshold(FLO, threshmin=np.median(FLO), newval=0)

            CLI = ContextIn.FF(FLO)
            CLO = ContextOut.FF(CLI)

        for item in data[str(i)]['analysis']:
            FLI = FeatureLayerIn.FF(item)
            FLI = stats.threshold(FLI, threshmin=np.median(FLI), newval=0)
            FLO = FeatureLayerOut.FF(FLI)
            FLO = stats.threshold(FLO, threshmin=np.median(FLO), newval=0)

            analysisData = np.concatenate((FLO, CLO), axis=0)
            sample.append(AnalysisLayer.FF(analysisData))

        for t in reversed(range(len(sample))):
            error = sample[t] - np.array(data[str(i)]['target'][t])

            dAL = AnalysisLayer.Grad(error, t)

            dFLOa = FeatureLayerOut.Grad(dAL[:3], t)
            FeatureLayerIn.Grad(dFLOa, t)

            for ts in reversed(range(len(data[str(i)]['context']))):
                dCLO = ContextOut.Grad(dAL[3:], ts)
                dCLI = ContextIn.Grad(dCLO, ts)

                dFLOc = FeatureLayerOut.Grad(dCLI, ts)
                FeatureLayerIn.Grad(dFLOc, ts)

            if epoch % display == 0:
                print(str(data[str(i)]['analysis'][t]) + " " + str(sample[t]) + " " + str(data[str(i)]['target'][t]))
        FeatureLayerIn.update(learning_rate=0.1)
        FeatureLayerOut.update(learning_rate=.1)
        ContextIn.update(learning_rate=.1)
        ContextOut.update(learning_rate=.1)
        AnalysisLayer.update(learning_rate=.1)

        if epoch % display == 0:
            print("Epoch: " + str(epoch))
            print()
    epoch += 1