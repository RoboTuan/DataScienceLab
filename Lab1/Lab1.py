#!/usr/bin/env python
# coding: utf-8

# ## Exercize 2.1

# In[ ]:


# 1

import csv

rows = []

with open("./iris.data") as f:
    for cols in csv.reader(f):
        if len(cols) == 5:
            tmp = []
            tmp = [float(cols[0]), float(cols[1]), float(cols[2]), float(cols[3]), cols[4]]
            rows.append(tmp)
            #print(tmp)
#print(rows)


# In[ ]:


# 2

from math import sqrt

means = [0, 0, 0, 0]
std_deviation = [0, 0, 0, 0]
measurements_len = len(rows)
print(measurements_len)

for row in rows:
    means[0] += row[0]
    means[1] += row[1]
    means[2] += row[2]
    means[3] += row[3]

print("Mean:")
means = list(map(lambda x: x/measurements_len, means))
print(means)
print("---------")

for row in rows:
    std_deviation[0] += (row[0] - means[0])**2
    std_deviation[1] += (row[1] - means[1])**2
    std_deviation[2] += (row[2] - means[2])**2
    std_deviation[3] += (row[3] - means[3])**2


print("Std deviation:")
std_deviation = list(map(lambda x: sqrt(x/measurements_len), std_deviation))
print(std_deviation)


# In[ ]:


# 3

iris_versicolor = list(filter(lambda x: x[4]=='Iris-versicolor', rows))
iris_setosa = list(filter(lambda x: x[4]=='Iris-setosa', rows))
iris_virginica = list(filter(lambda x: x[4]=='Iris-Virginica', rows))

versicolor_mean = [0, 0, 0, 0]
setosa_mean = [0, 0, 0, 0]
virginica_mean = [0, 0, 0, 0]

versicolor_std = [0, 0, 0, 0]
setosa_std = [0, 0, 0, 0]
virginica_std = [0, 0, 0, 0]

count = [0, 0, 0]

for row in rows:
    if row[4] == 'Iris-versicolor':
        versicolor_mean[0] += row[0]
        versicolor_mean[1] += row[1]
        versicolor_mean[2] += row[2]
        versicolor_mean[3] += row[3]
        
        count[0] += 1
            
    if row[4] == 'Iris-setosa':
        setosa_mean[0] += row[0]
        setosa_mean[1] += row[1]
        setosa_mean[2] += row[2]
        setosa_mean[3] += row[3]
        
        count[1] += 1
        
    if row[4] == 'Iris-virginica':
        virginica_mean[0] += row[0]
        virginica_mean[1] += row[1]
        virginica_mean[2] += row[2]
        virginica_mean[3] += row[3]
        
        count[2] += 1
        

print("Mean: ")
versicolor_mean = list(map(lambda x: x/count[0], versicolor_mean))
print(f"versicolor_mean: {versicolor_mean}")

setosa_mean = list(map(lambda x: x/count[1], setosa_mean))
print(f"setosa_mean: {setosa_mean}")

virginica_mean = list(map(lambda x: x/count[2], virginica_mean))
print(f"virginica_mean: {virginica_mean}")
print("------------")



for row in rows:
    if row[4] == 'Iris-versicolor':
        versicolor_std[0] += (row[0] - versicolor_mean[0])**2
        versicolor_std[1] += (row[1] - versicolor_mean[1])**2
        versicolor_std[2] += (row[2] - versicolor_mean[2])**2
        versicolor_std[3] += (row[3] - versicolor_mean[3])**2
            
    if row[4] == 'Iris-setosa':
        setosa_std[0] += (row[0] - setosa_mean[0])**2
        setosa_std[1] += (row[1] - setosa_mean[1])**2
        setosa_std[2] += (row[2] - setosa_mean[2])**2
        setosa_std[3] += (row[3] - setosa_mean[3])**2
        
    if row[4] == 'Iris-virginica':
        virginica_std[0] += (row[0] - virginica_mean[0])**2
        virginica_std[1] += (row[1] - virginica_mean[1])**2
        virginica_std[2] += (row[2] - virginica_mean[2])**2
        virginica_std[3] += (row[3] - virginica_mean[3])**2

        
print("Std deviation: ")
versicolor_std = list(map(lambda x: sqrt(x/count[0]), versicolor_std))
print(f"versicolor_std: {versicolor_std}")

setosa_std = list(map(lambda x: sqrt(x/count[1]), setosa_std))
print(f"setosa_std: {setosa_std}")

virginica_std = list(map(lambda x: sqrt(x/count[2]), virginica_std))
print(f"virginica_std: {virginica_std}")


# In[ ]:


# 4 optional #5 optional

test_iris = []

with open("./iris_test.csv") as f:
    for row in csv.reader(f):
        row = [float(x) for x in row]
        test_iris.append(row)

        
#print(test_iris)
test_len = len(test_iris)

'''
for i,row in enumerate(test_iris):
    std1 = [abs(x-y) for x,y in zip(row,versicolor_mean)]
    std1 = sum(std1)
    
    std2 = [abs(x-y) for x,y in zip(row,setosa_mean)]
    std2 = sum(std2)
    
    std3 = [abs(x-y) for x,y in zip(row,virginica_mean)]
    std3 = sum(std3)

    m = min(std1,std2,std3)
    
    if m == std1:
        print(f"The species of the row {i} is versicolor_mean")
    elif m == std2:
        print(f"The species of the row {i} is setosa_mean")
    else:
        print(f"The species of the row {i} is virginica_mean")
'''

e1 = [0] * test_len
e2 = [0] * test_len
e3 = [0] * test_len

def euclidean_distance(vect1,vect2):
    sum = 0
    
    for elem1, elem2 in zip(vect1, vect2):
        sum += (elem1-elem2)**2
        
    return sqrt(sum)

e1[0] = euclidean_distance(test_iris[0], versicolor_mean)
e2[0] = euclidean_distance(test_iris[0], setosa_mean)
e3[0] = euclidean_distance(test_iris[0], virginica_mean)

e1[1] = euclidean_distance(test_iris[1], versicolor_mean)
e2[1] = euclidean_distance(test_iris[1], setosa_mean)
e3[1] = euclidean_distance(test_iris[1], virginica_mean)

e1[2] = euclidean_distance(test_iris[2], versicolor_mean)
e2[2] = euclidean_distance(test_iris[2], setosa_mean)
e3[2] = euclidean_distance(test_iris[2], virginica_mean)



for i in range(test_len):
    euclidean = min(e1[i], e2[i], e3[i])

    if euclidean == e1[i]:
        print("the {i} row is versicolor")
    elif euclidean == e2[i]:
        print("the {i} row is setosa")
    else:
        print("the {i} row is virginica")





# ## Exercize 2.2

# In[ ]:


# 1 and #2

import json

dict = {}

with open("to-bike.json") as f:
    dict = json.load(f)
    #print(dict)


stations = dict['network']['stations']

counter = 0

for station in stations:
    if station['extra']['status'] == 'online':
        counter += 1

print(counter)
#print(stations)


# In[ ]:


# 3

n_free_bikes = 0
n_free_docks = 0

n_free_bikes = sum(station['free_bikes'] for station in stations)
n_free_docks = sum(station['empty_slots'] for station in stations)

print(f"Number of avaiable bikes: {n_free_bikes}")
print(f"Number of avaiable docks: {n_free_docks}")


# In[ ]:


# 4 optional
from math import cos, acos, sin

def distance_coords(lat1, lng1, lat2, lng2):
    """Compute the distance among two points."""
    deg2rad = lambda x: x * 3.141592 / 180
    lat1, lng1, lat2, lng2 = map(deg2rad, [ lat1, lng1, lat2, lng2 ])
    R = 6378100 # Radius of the Earth, in meters
    return R * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))

latitude, longitude = 45.074512, 7.694419 

distance_stations = []
minimum = [0, None]

for station in stations:
    distance = distance_coords(latitude, longitude, station['latitude'], station['longitude'])
    if station['free_bikes'] and (distance < minimum[0] or minimum[1] == None):
        minimum[0] = distance
        minimum[1] = station
    
#print(distance_stations)

print(f"The station with the minimum distance is: {minimum[1]['name']}",
      f"with a distance of {minimum[0]}")


# In[ ]:


def distance_from_point_2(dataset, lat, lng):
    v = [ (s, distance_coords(lat, lng, s["latitude"], s["longitude"])) for s in stations if s["free_bikes"] > 0 ]
    return min(v, key=lambda w: w[1])

station, distance = distance_from_point_2(stations, 45.074512, 7.694419)
print("Closest station:", station["name"])
print("Distance:", distance, "meters")
print("Number of available bikes:", station["free_bikes"])


# ## Execize 2.3

# In[28]:


# 1 and #2

digits = []

with open("./mnist_test.csv") as f:
    for row in csv.reader(f):
        digit = {'digit': row[0], 'values': row[1:]}
        digits.append(digit)

# Print values of first row
#print(digits[0]['values'][:])

    
def getDigit(data_set, number_of_row):
    matrix_length = int(sqrt(len(data_set[number_of_row]['values'])))
    printable = []
    start = 0
    finish = matrix_length
    
    printable = list(map(lambda x: " " if 0<=int(x)<64 else ("." if 64<=int(x)<128 else (
    "*" if 128<=int(x)<192 else "#")),data_set[number_of_row]['values']))
    
    # Print as a 28x28 matrix   
    for i in range(matrix_length):
        #printable[406] = 'w'    #for debug to see the max_difference, see #5 optional
        print(*printable[start:finish], sep = '')
        start += matrix_length
        finish += matrix_length
        


getDigit(digits,129)


# In[ ]:


# 3

def euclidean_dist(x, y):
    return sum([ (int(x_i) - int(y_i)) ** 2 for x_i, y_i in zip(x, y) ]) ** 0.5

positions = [ 25, 29, 31, 34 ]

for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        a = positions[i]
        b = positions[j]
        print(a, b, euclidean_dist(digits[a]['values'], digits[b]['values']))


# In[33]:


# 4 optional

values = [digits[i]['digit'] for i in positions]

print(values)


# In[56]:


# 5 optional
c_z = 0
c_o = 0

n = 784

c_o = sum(int(digit['digit']) == 1 for digit in digits)
print(f"Number of 1: {c_o}")

c_z = sum(int(digit['digit']) == 0 for digit in digits)
print(f"Number of 0: {c_z}")

zero = [0] * n
one = [0] * n

for digit in digits:
    if int(digit['digit']) == 0:
        for i, pixel in enumerate(digit['values']):
            if int(pixel) >= 128:
                zero[i] += 1
    elif int(digit['digit']) == 1:
        for i, pixel in enumerate(digit['values']):
            if int(pixel) >= 128:
                one[i] += 1
                

distances = [abs(zero[i] - one[i]) for i in range(n)]
                
'''
# enumerates the distances vector -> i, distance[i] and
# compares x[1] (x = (i, distance[i]) -> x[1] = distance[i]) and
# returns the index ([0] at the end) 

def argmax(w):
    return max(enumerate(w), key=lambda x: x[1])[0]
print(argmax(distances))
'''
    
max_distance = max(distances)
index_max_distance = distances.index(max_distance)

print(f"max_distance is: {max_distance}")
print(f"The index of max_distance is: {index_max_distance}")
print(f"The grid index of max_distance is: [{int(index_max_distance/28)}][{index_max_distance%28}]")

'''
# the number of ones and zeros isn't equal, it's better to normalize the differences
diff_norm = [ abs(z / Z_count - o / O_count) for z,o in zip (Z,O) ] argmax(diff_norm)
'''


# It's at the center of the grid because the symbol *1* have a line of black pixels passing from that point and *0* have an empty space
