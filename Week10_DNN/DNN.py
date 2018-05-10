#_*_ coding:utf-8 _*_

import numpy as np
import sys

# activation func - Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# activation func - relu
def relu(x):
    return np.maximum(0.0,x)

# derivate value of relu    
def relu_deri(x):
    deri = (x>0.0) # Boolean value
    return np.float32(deri)

# Predict XOR Answer
def predict(feat,label,weight1,weight2,weight3):
    correct = 0
    for i in range(len(feat)):
        L1 = np.dot(feat[i],weight1)
        L1 = relu(L1)
        L2 = np.dot(np.append(1,L1),weight2) 
        L2 = relu(L2)
        L3 = np.dot(np.append(1,L2),weight3)
        
        result = np.argmax(sigmoid(L3))
        
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
lr = 0.1

# 2 Layer Perceptron
w1 = np.random.random([3,5])
w2 = np.random.random([6,5])
w3 = np.random.random([6,2])

# If error_rate = 0, suc_error + 1
# If successive error reach 10, Finish training
suc_error=0

# epoch index
t=0

while(True):
    cost = 0
    
    
    # training data index shuffle
    idx = range(len(train_data))
    np.random.shuffle(idx)
        
    for i in range(len(train_data)):
        
        L1 = np.dot(train_data[idx[i]],w1) # (input * w1)
        
        L1_cal = relu(L1)
        L1 = np.append(1,L1_cal) # bias 삽입
        L2 = np.dot(L1,w2)                 # (Layer1 * w2)
        
        # activation
        L2_cal = relu(L2)
        L2 = np.append(1,L2_cal) # bias 삽입
        L3 = np.dot(L2,w3)                 # (Layer1 * w2)
            
        active = sigmoid(L3)
        # compute cost (accumulate & divide)
        cost += np.float32(np.sum((train_label[idx[i]]-active))**2)/2
        
        # weight tmp variable
        n_w1 = np.zeros([3,5],np.float32)
        n_w2 = np.zeros([6,5],np.float32)
        n_w3 = np.zeros([6,2],np.float32)
        
        
        # w3 renewal
        
        cost3 = train_label[idx[i]] - active
        n_w3 = w3 + lr*cost3*((active)*(1-active))*(L2.reshape([6,1]))

        
        # w2 renewal
        cost2 = np.zeros([5,1])

        # compute w2's cost
        for j in range(len(cost2)):
            cost2[j]=np.sum(cost3*(active*(1-active))*w3[j])
            
        cost2 = cost2.reshape([5,])
        n_w2 = w2 + lr*cost2*relu_deri(L2_cal)*L1.reshape([6,1])
        
       
        # w1 renewal
        cost1 = np.zeros([5,1])
        
        # compute w1's cost
        for j in range(len(cost1)):
            cost1[j]=np.sum(cost2*(relu_deri(L2_cal))*w2[j])
        
        cost1 = cost1.reshape([5,])
        n_w1 = w1 + lr*cost1*relu_deri(L1_cal)*train_data[idx[i]].reshape([3,1])
        
        
        # allocate new weight values
        w1 = n_w1
        w2 = n_w2
        w3 = n_w3
    
    print "\nepoch {} cost:{}".format(t+1,cost/len(train_data))
    print "error_rate:{}\n".format(1-predict(train_data,train_label,w1,w2,w3))
    correct_rate = predict(train_data,train_label,w1,w2,w3)
    
    fp.write("epoch {}\ncost : {}\n".format(t+1,cost/len(train_data)))
    fp.write("error_rate : {}\n\n".format(1-correct_rate))

    if correct_rate == 1.0:
        suc_error+=1
        if suc_error == 10:
            break;

    # epoch + 1
    t+=1
    if t == 10000:
        print "Please re-train"
        sys.exit()




    
print "End!"

fp.close()

# ---------- test -----------

fp = open("test_output.txt",'wt')


correct = 0

fp.write("\n2015920070 신권철\nTrain log\n\n")
for i in range(4):
    L1 = np.dot(train_data[i],w1)
    L1 = relu(L1)
    L2 = np.dot(np.append(1,L1),w2)
    L2 = relu(L2)
    L3 = np.dot(np.append(1,L2),w3)
    
    result = np.argmax(sigmoid(L3))
    
    if train_label[i][result] == 1:
        correct += 1
        fp.write("---correct---\n")
    else:
        fp.write("***wrong***\n")
        
    fp.write("Data : [{},{}]\nAnswer : [{}]\n\n".format(train_data[i][1],train_data[i][2],result))

fp.write("Error rate : %.3f"%(1-np.float32(correct)/4))
print "Test Error rate : %.3f"%(1-np.float32(correct)/4)
    
fp.close()


