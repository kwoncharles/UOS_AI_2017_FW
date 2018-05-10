#_*_ coding:utf-8 _*_

import numpy as np

# activation func - Sigmoid
def activation(x):
    return 1/(1+np.exp(-x))

# Predict XOR Answer
def predict(feat,label,weight1,weight2):
    correct = 0
    for i in range(len(feat)):
        L1 = np.dot(feat[i],weight1)
        L1 = activation(L1)
        L2 = np.dot([1.0,L1[0],L1[1],L1[2]],weight2)
        result = np.argmax(activation(L2))
        
        if label[i][result] == 1:
            correct = correct+1
        else:
            print 'false : {}'.format(feat[i])

    correct_rate = np.float32(correct)/len(feat)
    return correct_rate


fp = open("train_log.txt",'wt')

# Data & Label

# [bias,feat1,feat2]
train_data = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
train_label = np.array([[1,0],[0,1],[0,1],[1,0]])

# Hyper parameters
lr = 1
epoch=1000

# 2 Layer Perceptron
w1 = np.random.random([3,3])*0.01
w2 = np.random.random([4,2])


for t in range(epoch):
    cost = 0
    
    # training data index shuffle
    idx = range(len(train_data))
    np.random.shuffle(idx)
    
    for i in range(len(train_data)):
        
        L1 = np.dot(train_data[idx[i]],w1) # (input * w1)
        
        L1_cal = activation(L1)
        L1 = np.array([1,L1_cal[0],L1_cal[1],L1_cal[2]])
        L2 = np.dot(L1,w2)                 # (Layer1 * w2)
        
        # activation
        active = activation(L2)
        
        # compute cost (accumulate & divide)
        cost += np.float32(np.sum((train_label[idx[i]]-active))**2)/2
        
        # weight tmp variable
        n_w1 = np.zeros([3,3],np.float32)
        n_w2 = np.zeros([4,2],np.float32)
        
        temp = (train_label[idx[i]] - active)*(active)*(1-active)
        
        # w2 renewal
        n_w2 = w2 + lr*temp*(L1.reshape([4,1]))
        
        
        # w1 renewal
        w1_1 = np.sum(temp*w2[1])
        w1_2 = np.sum(temp*w2[2])
        w1_3 = np.sum(temp*w2[3])
        temp_w1 = np.array([w1_1,w1_2,w1_3])
        
        n_w1 = w1 + (lr*temp_w1*(train_data[idx[i]].reshape([3,1])*L1_cal*(1-L1_cal)))
        
        # allocate new weight values
        w1 = n_w1
        w2 = n_w2
    
    print "\nepoch {} cost:{}".format(t+1,cost/len(train_data))
    print "error_rate:{}\n".format(1-predict(train_data,train_label,w1,w2))

    correct_rate = predict(train_data,train_label,w1,w2)
    
    fp.write("epoch {}\ncost : {}\n".format(t+1,cost/len(train_data)))
    fp.write("error_rate : {}\n\n".format(1-correct_rate))

print "End!"

fp.close()

# ---------- test -----------

fp = open("test_output.txt",'wt')

correct = 0

fp.write("\n2015920070 신권철\nTrain log\n\n")
for i in range(4):
    L1 = np.dot(train_data[i],w1)
    L1 = activation(L1)
    L2 = np.dot([1.0,L1[0],L1[1],L1[2]],w2)
    result = np.argmax(activation(L2))
    
    if train_label[i][result] == 1:
        correct += 1
        fp.write("---correct---\n")
    else:
        fp.write("***wrong***\n")
    
    fp.write("Data : [{},{}]\nAnswer : [{}]\n\n".format(train_data[i][1],train_data[i][2],result))

fp.write("Error rate : %f"%(1-np.float32(correct)/4))
print "Test Error rate : %f"%(1-np.float32(correct)/4)

fp.close()


