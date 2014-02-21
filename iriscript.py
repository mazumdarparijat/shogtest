from numpy import array, concatenate, ones, zeros, nonzero, dot
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

# wrap to Shogun features
train_features = RealFeatures(obsmatrix)

train_features_clone = train_features
submean = PruneVarSubMean(False)
submean.init(train_features_clone)
submean.apply_to_feature_matrix(train_features_clone)
preprocessor = PCA()
preprocessor.set_target_dim(2)
print preprocessor.get_target_dim()
preprocessor.init(train_features_clone)
pca_transform = preprocessor.get_transformation_matrix()
print pca_transform
new_features = dot(pca_transform.T, train_features)
print new_features













# number of cluster centers = 3
k = 3

# distance function features - euclidean
distance = EuclideanDistance(train_features, train_features)

# initialize KMeans object
kmeans = KMeans(k, distance)

# use kmeans++ to initialize centers [play around: change it to False and compare results]
kmeans.set_use_kmeanspp(True)
# training kmeans
kmeans.train(train_features)

# labels for data points
result = kmeans.apply()

labels = concatenate((zeros(50),ones(50),2.*ones(50)),1)
ground_truth = MulticlassLabels(array(labels,dtype='float64'))

AccuracyEval = ClusteringAccuracy()
AccuracyEval.best_map(result, ground_truth)

compare = result.get_labels()-labels
diff = nonzero(compare)

figure,axis = pyplot.subplots(1,1)
axis.plot(obsmatrix[2,:],obsmatrix[3,:],'x',color='black', markersize=5)
axis.plot(obsmatrix[2,diff],obsmatrix[3,diff],'ko',color='r', markersize=7)
axis.set_xlim(-1,8)
axis.set_ylim(-1,3)
axis.set_title('Difference')
pyplot.show()

print AccuracyEval.evaluate(result, ground_truth)

		

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

