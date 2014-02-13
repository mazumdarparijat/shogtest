from numpy import array as arr
import matplotlib.pyplot as pyplot

f = open('iris.data')
features = []
labels = []
for line in f:
	words = line.rstrip().split(',')
	features.append([float(i) for i in words[0:4]])
	labels.append(words[4])	

obsmat = arr(features).T
figure,axis = pyplot.subplots(1,1)
axis.plot(obsmat[2,0:49], obsmat[3,0:49], 'o', color= 'green', markersize=5)
axis.plot(obsmat[2,50:99], obsmat[3,50:99], 'o', color= 'red', markersize=5)
axis.plot(obsmat[2,100:149], obsmat[3,100:149], 'o', color= 'blue', markersize=5)
axis.set_xlim(-1,8)
axis.set_ylim(-1,3)
pyplot.show()
f.close()


