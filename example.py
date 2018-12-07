## plotting example 12.8 from MMD
## APM120, Eli Tziperman, 201709
import numpy as np
from numpy import linalg
import scipy as scipy
import matplotlib.pyplot as plt
import matplotlib

# Training data and labels:
X=np.array([[1,  3,  2,  4],
            [2,  4,  1,  3]]);
y= np.array([1,  1, -1, -1]);
N=np.size(y);

plt.figure(1); plt.clf();

# plot SVM lines:
u=-1;
v=1;
x=np.array([0,5]);
y1=-u*x/v+1/v;
y2=-u*x/v-1/v;
plt.plot(x,y1,'r-')
plt.plot(x,y2,'b-')
plt.legend(['$\mathbf{w}\cdot\mathbf{x}=ux+vy=1$' \
                ,'$\mathbf{w}\cdot\mathbf{x}=ux+vy=-1$']);

# plot training data:
for i in range(0,N):
    if y[i]==1:
        plt.plot(X[0,i],X[1,i],'r+');
    if y[i]==-1:
        plt.plot(X[0,i],X[1,i],'bo');

plt.xlim([0, 5])
plt.ylim([0, 5])
plt.xlabel('u')
plt.ylabel('v')
plt.title('SVMs example 12.8 from MMD')
plt.show();