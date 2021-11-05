import numpy as np

#Import scikitlearn for machine learning functionalities
import sklearn
from sklearn.manifold import TSNE 

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
#%matplotlib inline
from os import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(123)
import seaborn as sb

data1 = np.loadtxt('train.dat')
data2 = np.loadtxt('test.dat')
data3 = np.loadtxt('cov.dat')

#X1=data1[:,[0,2,3,8,9,12,13,14,16,19,23,29]]
#X1=data1[:,[0,2,4,8,12]]
#X2=data2[:,[0,2,4,8,12]]
X3=data3[:,[0,2,4,8,12]]
#print(X)

#x1 = TSNE(perplexity=30,random_state=1).fit_transform(X1)
#x2 = TSNE(perplexity=30,random_state=1).fit_transform(X2)
x3 = TSNE(perplexity=30,random_state=1).fit_transform(X3)

#cols=np.column_stack((x1,x2,x3))
#np.savetxt("chiz_test_2d.log", x2)
#np.savetxt("chiz_cov_2d.log", x3)


# Create a scatter plot.
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
ax.tick_params(axis='both',which='major', labelsize=22)
#ax.scatter(x1[:,0], x1[:,1], lw=0, s=40, c='blue')#, c=palette[colors.astype(np.int)])
#ax.scatter(x2[:,0], x2[:,1], lw=0, s=40, c='green')
ax.scatter(x3[:,0], x3[:,1], lw=0, s=40, c='red')

ax.set_xlim([-15,15])
ax.set_ylim([-15,15])

#ax.set_xticks(np.arange(-15,15,5))
#ax.set_yticks(np.arange(-15,15,5))

plt.show()
