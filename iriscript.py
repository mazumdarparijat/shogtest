f = open('iris.data')
data=[]
for line in f:
	data.append(line.rstrip().split(','))
print data	
