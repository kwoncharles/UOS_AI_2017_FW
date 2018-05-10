#_*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np
import random


# Classify input data with input weight
def getClass(x, param):
    # True is salmon, False is seabass
    
    if (param[0]*x[0] + param[1]*x[1] + param[2]) <= 0:
        return True
    else:
        return False

# For when graph's orientation is switched (do so above does)
def getClassN(x, param):
    if (param[0]*x[0] + param[1]*x[1] + param[2]) > 0:
        return True
    else:
        return False

# Compute cost of current parameters
def getCost(sal,seab,weight):
    correct1 = 0
    correct2 = 0
    
    # Classify seabass and salmon with current parameters. And save the wrong counts
    for i in range(len(sal)):
        if getClass(sal[i], weight) == True:
            correct1 += 1

    for i in range(len(seab)):
        
        if getClass(seab[i], weight) == False:
            correct2 += 1
        
        cost = 100 - (correct1 + correct2)
    
        # When cost exceeds 50, graph's orientation is switched
        # Re-compute the cost with getClassN func.
    if (cost > 50):
        correct1 = 0
        correct2 = 0
        for i in range(len(sal)):
            if getClassN(sal[i], weight) == True:
                correct1 += 1
        
        for i in range(len(seab)):
            if getClassN(seab[i], weight) == False:
                correct2 += 1

        cost = 100 - (correct1 + correct2)

    return cost

# Open Input data

fp = open("salmon_train.txt","r")
lines = fp.readlines()

fp = open("seabass_train.txt","r")
lines2 = fp.readlines()

fp.close()

salmon = list()
seabass = list()

for line in lines:
    salmon.append(((line.split())))

for line in lines2:
    seabass.append(((line.split())))

for i in range(len(salmon)):
    salmon[i] = map(float,salmon[i])

for i in range(len(seabass)):
    seabass[i] = map(float,seabass[i])

print  'Training starts'

fp = open("train_log",'w')

# Initialize parameters
T = 100
weight = [2.0, -1.0, -180.0]
alpha = 0.99

# Variables to store optimal parameters
min_cost = 100
min_weight = weight

count = 0

salmon_num = len(salmon)
seabass_num = len(seabass)
fish_num = salmon_num + seabass_num

while(T > 0.001):
    
    # Random values that add to new weight
    # 새로운 weight값에 더해질 값들을 랜덤으로 생성
    rnum1 = random.uniform(-0.01, 0.01)
    rnum2 = random.uniform(-0.01, 0.01)
    rnum3 = random.uniform(-10.0, 10.0)
    
    # Back up previous weight
    b_weight = weight
    
    # Save new weight in temp_weight
    temp_weight = [weight[0] + rnum1, weight[1] + rnum2, weight[2] + rnum3]
    
    # Compute cost to compare new and previous weights
    b_cost = getCost(salmon,seabass,weight)
    c_cost = getCost(salmon,seabass,temp_weight)
    
    E = c_cost - b_cost
    
    count+=1
    
    printit = "{}. cost = {}, weight = {}\n".format(count, c_cost, temp_weight)
    print printit
    fp.write(printit)
    
    # If previous cost is bigger, than renew the weight
    if ( E < 0 ) :
        weight = temp_weight
        
        # If new cost is smaller than current min_cost, new the min_cost
        if (c_cost < min_cost):
            min_cost = c_cost
            min_weight = temp_weight

    # Else renew with a contant probability
    else :
        rnum = random.uniform(0.0,1.0)
        if rnum < np.exp(-E / T):
            weight = temp_weight
    
    
    
    if (E > 0):
        print "New cost is higher than before\n\n"
    elif (E == 0):
        print "New cost is same with before\n\n"
    else:
        print "New cost is less than before\n\n"
    
    # reduce T every step
    T = alpha * T

print 'Training ends'
fp.close()



fp = open("salmon_test.txt","r")
lines = fp.readlines()

fp = open("seabass_test.txt","r")
lines2 = fp.readlines()

fp.close()

t_salmon = list()
t_seabass = list()

for line in lines:
    t_salmon.append(((line.split())))

for line in lines2:
    t_seabass.append(((line.split())))

for i in range(len(t_salmon)):
    t_salmon[i] = map(float,t_salmon[i])

for i in range(len(t_seabass)):
    t_seabass[i] = map(float,t_seabass[i])


test_weight = min_weight

salmon_cor = []
salmon_incor = []
seabass_cor = []
seabass_incor = []

for i in range(len(t_salmon)):
    if (getClass(t_salmon[i],test_weight)==True) :
        salmon_cor.append(i)
    else :
        salmon_incor.append(i)

for i in range(len(t_seabass)):
    if (getClass(t_seabass[i],test_weight)==False) :
        seabass_cor.append(i)
    else :
        seabass_incor.append(i)


# Plot graph

_, ax = plt.subplots()

xList = []
yList = []
for data in salmon_cor:
    x, y = t_salmon[data]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'b^', Label = 'salmon_cor')

xList = []
yList = []
for data in salmon_incor:
    x, y = t_salmon[data]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'rs', Label = 'salmon_incor')

xList = []
yList = []
for data in seabass_cor:
    x, y = t_seabass[data]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'g^', Label = 'seabass_cor')

xList = []
yList = []
for data in seabass_incor:
    x, y = t_seabass[data]
    xList.append(x)
    yList.append(y)
ax.plot(xList, yList, 'ro', Label = 'seabass_incor')

ax.grid(True)
ax.legend(loc='upper right')

ax.set_xlabel('Length of Body')
ax.set_ylabel('Length of Tail')

ax.set_xlim((None,None))
ax.set_ylim((None,None))

plt.savefig('fish1.png')
plt.show() # Red dots are wrong value ( Square is salmon, Circle is seabass.)

