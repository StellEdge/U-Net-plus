# 根据.npy生成灰度图.png

import scipy.misc
import numpy as np

for i in range(5):
    a = np.load('Results/npy_SGD/' + str(i) + '.npy')
    scipy.misc.imsave(str(i) + ".png", a)
