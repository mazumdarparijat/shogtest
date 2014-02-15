from numpy import array
import matplotlib.pyplot as pyplot
from modshogun import *
f = open('iris.data')
features = []
# read data from file
for line in f:
	words = line.rstrip().split(',')
	features.append([float(i) for i in words[0:4]])

f.close()

# create observation matrix
obsmatrix = array(features).T

# plot the data
figure,axis = pyplot.subplots(1,1)
# First 50 data belong to Iris Sentosa, plotted in green
axis.plot(obsmatrix[2,0:49], obsmatrix[3,0:49], 'o', color='green', markersize=5)
# Next 50 data belong to Iris Versicolour, plotted in red
axis.plot(obsmatrix[2,50:99], obsmatrix[3,50:99], 'o', color='red', markersize=5)
# Last 50 data belong to Iris Virginica, plotted in blue
axis.plot(obsmatrix[2,100:149], obsmatrix[3,100:149], 'o', color='blue', markersize=5)
axis.set_xlim(-1,8)
axis.set_ylim(-1,3)
axis.set_title('3 varieties of Iris plants')
pyplot.show()

# wrap to Shogun features
train_features = RealFeatures(obsmatrix)

# number of cluster centers = 3
k = 3

# distance function features - euclidean
distance = EuclideanDistance(train_features, train_features)

# initialize KMeans object
kmeans = KMeans(k, distance)

# use kmeans++ to initialize centers [play around: change it to False and compare results]
kmeans.set_use_kmeanspp(True)

# training method is Lloyd by default [play around: change it to mini-batch by uncommenting the following lines]
#kmeans.set_train_method(KMM_MINI_BATCH)
#kmeans.set_mbKMeans_params(20,30)

# training kmeans
kmeans.train(train_features)

# labels for data points
result = kmeans.apply()



# plot the clusters over the original points in 2 dimensions
figure,axis = pyplot.subplots(1,1)
for i in xrange(150):
    if result[i]==0.0:
        axis.plot(obsmatrix[2,i],obsmatrix[3,i],'ko',color='r', markersize=5)
    elif result[i]==1.0:
        axis.plot(obsmatrix[2,i],obsmatrix[3,i],'ko',color='g', markersize=5)
    else:
        axis.plot(obsmatrix[2,i],obsmatrix[3,i],'ko',color='b', markersize=5)

axis.set_xlim(-1,8)
axis.set_ylim(-1,3)
axis.set_title('Iris plants clustered based on attributes')
pyplot.show()

