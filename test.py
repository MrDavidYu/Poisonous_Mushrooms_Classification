#!/usr/bin/python
# -*- coding: utf-8 -*-

# Using LabelEncoder to transform letters to numbers, for example, a,b,c may be changed to 1,2,3.....
# save the result in the txt file.
# I have fininshed the transform, the result files are data.txt, data_label.txt, the final result files are
# result.csv and result_label.csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
file = open('data_label.txt', 'w') #Features: data.txt; result.csv. Labels: data_label.txt; result_label.csv

for num in range(0,1):  # Modifing this range. (0,1) means to transform the label of mushrooms, (1,22) means
                        # to transform the features of mushrooms, but remember to change the filename to store
                        # the features or labels. Features: data.txt; result.csv. Labels: data_label.txt; result_label.csv
    data=pd.read_csv('./../mushrooms.csv',usecols=[num])
    data = data.values
    data = data.ravel()
    print data
    le = preprocessing.LabelEncoder()
    le.fit(data)
    result = le.transform(data)
    r = ",".join(str(e) for e in result)
    file.write(r + "\n");
    #np.savetxt("data.txt", result)


    print r
file.close();

import csv
with open('result_label.csv', 'wb') as csvfile:  #Features: data.txt; result.csv. Labels: data_label.txt; result_label.csv
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open('data_label.txt', 'rb') as filein: #Features: data.txt; result.csv. Labels: data_label.txt; result_label.csv
        for line in filein:
            line_list = line.strip('\n').split(',')
            spamwriter.writerow(line_list)