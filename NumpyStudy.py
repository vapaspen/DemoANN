import numpy as np
from ANN import ANN
from RNN import RNN


rng = np.random.RandomState(np.random.randint(low=10, high=10000))
LayerIn = ANN(Xc=2 , Hc=3, rng=rng)
LayerOut = RNN(Xc=3 , Hc=2, rng=rng)

data = [[0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
        ]
target = [[1, 0],
          [0, 1],
          [1, 1],
          [0, 0]
          ]

display = 1000
epoch =0
while epoch < 10000:
    LayerIn.reset()
    LayerOut.reset()
    sample = []
    for i in data:
        LIn = LayerIn.FF(i)
        sample.append(LayerOut.FF(LIn))

    for t in reversed(range(len(sample))):
        error = sample[t] - np.array(target[t])

        LOut = LayerOut.Grad(error, (t))
        LayerIn.Grad(LOut, (t))
        if epoch % display == 0:
            print(str(data[t]) + " " + str(sample[t]) + " " + str(target[t]))
    LayerIn.update(learning_rate=1)
    LayerOut.update(learning_rate=1)

    if epoch % display == 0:
        print(sample[0])
    epoch += 1
LayerIn.save('SavedLayers/LayerIn')
LayerOut.save('SavedLayers/LayerOut')