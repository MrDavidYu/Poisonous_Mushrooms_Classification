#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import kmedoids
import re
mushrooms = 8000  # Change it to run on smaller dataset
po = 0
ea = 0
lists = []
lists2 = []
list3 = []
temp = []
list4 = []
list5 = []
# Read the data from result.csv and result_label.csv column by column.
# Change the data format to a list like [[1,2,3,4,5...],[2,3,4,5,6.....].......]

# Can run directly with the result.csv and result_label.csv at the same directory
# If need to run on a smaller dataset, just changing the variate mushrooms above to smaller number.
for num in range(0,mushrooms):
    data=pd.read_csv('./result.csv', header= None, usecols=[num])
    data = data.values

    data = data.ravel()
    data1 = str(data)
    data1 = data1.strip("[]")
    data1 = re.sub(r'\s+', ' ', data1)
    data1 = data1.lstrip(' ')
    list3 = data1.split(" ")
    lists = []
    for xx in list3:
        ab = int(xx)
        lists.append(ab)
    list4.append(lists)

    #print list4

    data = pd.read_csv('./result_label.csv', header=None, usecols=[num])
    data = data.values

    data = data.ravel()
    data1 = str(data)
    data1 = data1.strip("[]")
    data1 = re.sub(r'\s+', ' ', data1)
    data1 = data1.lstrip(' ')
    list3 = data1.split(" ")
    lists = []
    for xx in list3:
        ab = int(xx)
        lists.append(ab)
    list5.append(lists)

    #print list5






import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Initialize PCA
pca = PCA(n_components=2)
data2 = np.array(list4)


# distance matrix
D = pairwise_distances(data2, metric='euclidean')

# split into 2 clusters
M, C = kmedoids.kMedoids(D, 2)
newData = pca.fit_transform(list4)

#for point_idx in M:
    #print( list4[point_idx] )


presult = 0
for label in C:
    for point_idx in C[label]:
        #print('label {0}:　{1}'.format(label, list4[point_idx]))


        if label == 0:
            plt.plot(newData[point_idx][0], newData[point_idx][1], 'r.')  # draw this point
            ea = ea + 1  # Calculate the sum of edible(poison) mushrooms
        if label == 1:
            plt.plot(newData[point_idx][0], newData[point_idx][1], 'go')  # draw this point
            po = po + 1  # Calculate the sum of poison(edible) mushrooms
        #print list4[point_idx][0]
        # Calculate the accuracy
        for xx in range(0,mushrooms):
            mark = 0
            for yy in range(0,21):
                if list4[point_idx][yy] != list4[xx][yy]:
                    mark = 1
                    break
            if mark == 0:

                if(label == list5[xx][0]):   # find one
                    presult = presult + 1
                break

print ('Accuracy: {0}%'.format(presult*100/mushrooms))
print "poison: "
print po
print "edible:"
print ea
plt.show()














