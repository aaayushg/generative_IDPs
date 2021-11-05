import numpy as np
data1 = np.loadtxt('ab40_encoded_training_set.dat')
data2 = np.loadtxt('ab40_encoded_test_set.dat')
data3 = np.loadtxt('cov_set.dat')

x1 = data1[:,0]
x2 = data2[:,0]
x3 = data3[:,0]
y1 = data1[:,27]
y2 = data2[:,27]
y3 = data3[:,27]

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,8))
# Creating plot

ax1.hist2d(x1, y1, range=np.array([(0,2),(0,2)]))
ax2.hist2d(x2, y2, range=np.array([(0,2),(0,2)]))
ax3.hist2d(x3, y3, range=np.array([(0,2),(0,2)]))

ax1.tick_params(axis='both',which='major', labelsize=16)
ax2.tick_params(labelsize=16)
ax3.tick_params(labelsize=16)

#ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)
#ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
#ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)

#ax1.set_xticklabels(np.arange(0,2,0.25),rotation=90)
#ax2.set_xticklabels(np.arange(0,2,0.25),rotation=90)
#ax3.set_xticklabels(np.arange(0,2,0.25),rotation=90)

ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)

#plt.title("Encoded Training 2D Histogram", fontsize=20)
#plt.title("Multivariate 2D Histogram", fontsize=20)
#fig.colorbar()
  
# show plot
plt.show()
