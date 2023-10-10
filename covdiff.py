import numpy as np
import matplotlib.pyplot as plt

#各取500个标准正态分布的数据
x1=np.random.normal(0,1,500)
x2=np.random.normal(0,1,500)

#将x1，x2以向量的形式排列起来
X=np.vstack((x1,x2)).T


#计算协方差
def cov(x1,x2):
    x1mean,x2mean=x1.mean(),x2.mean()
    Sigma=np.sum((x1-x1mean)*(x2-x2mean))/(len(x1)-1)
    return Sigma

#协方差矩阵
def covMatrix(X):
    matrix=np.array([[cov(X[0],X[0]),cov(X[0],X[1])],[cov(X[1],X[0]),cov(X[1],X[1])]])
    return matrix

covMatrix(X)

#output
'''
array([[0.00437243, 0.0433571 ],
       [0.0433571 , 0.42993022]])
'''


### 解法2
a = np.array([[1,2,3],[3,1,1]])
print(np.cov(a))

# output
'''
array([[1,-1],
       [-1,1.33]])
       '''