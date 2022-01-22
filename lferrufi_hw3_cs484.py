'''Luis Ferrufino
CS 484-000
HW#3
G#00997076
3/23/20'''

import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
data1 = genfromtxt('./dataset1.csv')
data2 = genfromtxt('./dataset2.csv')
from math import sqrt

myData = data1
#myData = data2

#BEGIN KMEANS SECTION:
'''
for k in range(2, 6): # repeat for k = 2 to 5

    kmeans = KMeans(n_clusters=k, random_state=0).fit(myData)    
    #now to compute the total sse for this value of k:
    
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    sse = 0

    for i in range(1, k + 1): # repeat for each cluster:

        for j in range(0, labels.size): # repeat for each member of the cluster:

            if ( labels[j] == i ):

                for l in range(0, 5): # repeat for each dimension:

                    sse += ( centroids[i][l] - myData[j][l] ) ** 2
        
    #report the value of sse:
    print("The SSE for k = " + str(k) + " is equal to " + str(sse))
'''
#END KMEANS SECTION.

#BEGIN DBSCAN SECTION:
'''
dbscan = DBSCAN(eps=0.3).fit(myData)
labels = dbscan.labels_
print(str(labels))
numClusters = len(set(labels)) - ( 1 if -1 in labels else 0 )
centroids = np.zeros((numClusters, 5))
numInCluster = np.zeros((numClusters, 1))
sse = 0

#calculate centroids of clusters:

for i in range(0, numClusters):

    for j in range(0, labels.size):

        if ( labels[j] == i ):

            numInCluster[i] += 1

            for l in range(0, 5):

                centroids[i][l] += myData[j][l]

for i in range(0, numClusters):

    for l in range(0, 5):

        centroids[i][l] /= numInCluster[i]

#calculate the sse:

for i in range(0, numClusters):

    for j in range(0, labels.size):

        if ( labels[j] == i ):

            for l in range(0, 5):

                sse += ( centroids[i][l] - myData[j][l] ) ** 2
print("The SSE for DBSCAN is " + str(sse))
'''
#END DBSCAN SECTION.

#BEGIN CORRELATION SECTION:
'''
#first, create a distance matrix:

#kmeans = KMeans(n_clusters=2, random_state=0).fit(myData)
#labels = kmeans.labels_
dbscan = DBSCAN(eps=0.3).fit(myData)
labels = dbscan.labels_
dist = np.zeros((labels.size, labels.size))

for i in range(0, labels.size):

    for j in range(0, labels.size):

        for l in range(0, 5):

            dist[i][j] += ( myData[i][l] - myData[j][l] ) ** 2

        dist[i][j] = sqrt(dist[i][j])

#second, create an incidence matrix:

inc = np.zeros((labels.size, labels.size))

for i in range(0, labels.size):

    for j in range(0, labels.size):

        if ( labels[i] == labels[j] ):

            inc[i][j] = 1 # 1 means incidence, 0 means not in the same cluster

#third, linearize both matrices:

distLin = np.reshape(dist, labels.size * labels.size)
incLin = np.reshape(inc, labels.size * labels.size)

#finally, calculate the correlation:

rho = np.cov(distLin, incLin, bias=True)[0][1] / ( np.std(distLin) * np.std(incLin) )
print("The correlation is rho = " + str(rho))
'''
#END CORRELATION SECTION.

#BEGIN SILHOUETTE SECTION:

#k=2
#kmeans = KMeans(n_clusters=k, random_state=0).fit(myData)
#labels = kmeans.labels_
dbscan = DBSCAN(eps=2).fit(myData)
labels = dbscan.labels_
k=len(set(labels)) - ( 1 if -1 in labels else 0 )

sil = np.zeros(labels.size) #will store the silhouette coefficients for each piont in myData

for i in range(0, labels.size): # for each point

    a = 0
    aCount = 0
    b = np.zeros(k)
    bCount = np.zeros(k) #for kmeans
    #bCount = np.zeros(k) #for dbscan
    for j in range(0, labels.size):

        if ( labels[i] == labels[j] ): # adding up intra-cluster distances

            aCount += 1
            temp = 0

            for l in range(0, 5):

                temp += ( myData[i][l] - myData[j][l] ) ** 2

            a += sqrt(temp)
        else: # adding up inter-cluster distances

            bCount[labels[j]] += 1

            temp = 0

            for l in range(0, 5):

                temp =+ ( myData[i][l] - myData[j][l] ) ** 2
            b[labels[j]] += sqrt(temp)
    a = a / aCount

    for j in range(0, k):

        if ( j != labels[i] ) :
        
            b[j] /= bCount[j]
    bMin = np.amin(np.delete(b, labels[i]))
    sil[i] = ( bMin - a ) / max(a, bMin)

print("The silhouette coefficient is s = " + str(np.average(sil)) )

#END SILHOUETTE SECTION.
