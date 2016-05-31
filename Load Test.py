from RNN import RNN
from ANN import ANN
import numpy as np

rng = np.random.RandomState(53489)
LayerIn = ANN(Xc=2 , Hc=4, rng=rng)
LayerOut = RNN(Xc=4 , Hc=2, rng=rng)

data = [[1, 1],
        [0, 0],
        [1, 0],
        [0, 1]
        ]

LayerIn.load('SavedLayers/LayerIn.npy')
LayerOut.load('SavedLayers/LayerOut.npy')



LayerIn.reset()
LayerOut.reset()
sample = []
for i in data:
    LIn = LayerIn.FF(i)
    sample.append(LayerOut.FF(LIn))

print(sample)