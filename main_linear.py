import numpy as np
import random
import matplotlib.pyplot as plt
from maml_linear_regression import maml_linear_regression
from data_generator import Data_Generator

data_generator=Data_Generator()
dim=10
real_theta=np.random.rand(dim)
theta,datas,outputs=data_generator.generate_linear(real_theta)

maml=maml_linear_regression(10,False) #タスク数、データ数(False数の場合10-1000のデータをランダムに選択)
#X,y,theta=maml.datas[0]
#print(maml.meta_gradient(X,y,theta))
print(maml.fit())
print(maml.real_theta)




x=np.arange(len(maml.process))
plt.plot(x,maml.test_process,label='test set')
plt.plot(x,maml.process,label='val set')
plt.plot(x,maml.real_process)
plt.legend(loc='upper right')
plt.ylim(0,1000)
plt.show()
"""
