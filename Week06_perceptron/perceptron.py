# _*_ coding: utf-8 _*_

import numpy as np
import random
import sys
import os.path as op

def LoadFish(file1, file2):
    fp = open(file1,'r')
    lines = fp.readlines()
    
    fp = open(file2,'r')
    lines2 = fp.readlines()
    
    fp.close()
    
    salmon = []
    seabass = []
    
    for line in lines:
        salmon.append(line.split())
    for line in lines2:
        seabass.append(line.split())
    
    # Convert into float
    for i in range(len(salmon)):
        salmon[i] = map(float,salmon[i])
    for i in range(len(seabass)):
        seabass[i] = map(float,seabass[i])
    
    # Insert 1.0 in Idx 0 of every data
    # To multiply with bias
    for i in range(len(salmon)):
        salmon[i] = [1.0,salmon[i][0],salmon[i][1]]
    for i in range(len(seabass)):
        seabass[i] = [1.0,seabass[i][0],seabass[i][1]]
    
    return salmon,seabass

class perceptron:
    
    
    def __init__(self,lr):
        # learning rate
        self.lr = lr
    
    # If w*x+b is higher than 0 then salmon, else seabass.
    def predict(self,weight,feat):
        if(weight[0]*feat[0] + weight[1]*feat[1] + weight[2]*feat[2] > 0):
            return 1 # salmon
        else:
            return 0 # seabass

    # Return correct answers' count
    def predict_all(self,weight,salmon,seabass):
        correct = 0
        
        for i in range(len(salmon)):
            if(self.predict(weight,salmon[i]) == 1):
                correct+=1
        for i in range(len(seabass)):
            if(self.predict(weight,seabass[i]) == 0):
                correct+=1
        
        return correct


    # Training method
    def train(self,feat1,feat2):
        
        # Initialize the weight
        weight = [random.uniform(-20,20),random.uniform(-20,20),random.uniform(0,50)]
        
        # Every loop, call predict_all method with current weight
        # When its result is the best this far, renew best variables
        b_weight = [0,0,0]
        b_rate = 0
        
        count = 0
        
        fp=open("train_log_[{}].txt".format(self.lr),'w')
        print "Training starts!\nTraining result file : train_log_[{}].txt\n".format(self.lr)
        print "Initial weight's number of correct answer : {}\n".format(self.predict_all(weight,feat1,feat2))

        while(True):
            for i in range(len(feat1)):
             
                n_weight = [0,0,0]
                
                n_weight[0] = weight[0] + self.lr*(1-self.predict(weight,feat1[i]))*feat1[i][0]
                n_weight[1] = weight[1] + self.lr*(1-self.predict(weight,feat1[i]))*feat1[i][1]
                n_weight[2] = weight[2] + self.lr*(1-self.predict(weight,feat1[i]))*feat1[i][2]
                
                weight = n_weight
            for i in range(len(feat2)):
                n_weight = [0,0,0]
                
                n_weight[0] = weight[0] + self.lr*(0-self.predict(weight,feat2[i]))*feat2[i][0]
                n_weight[1] = weight[1] + self.lr*(0-self.predict(weight,feat2[i]))*feat2[i][1]
                n_weight[2] = weight[2] + self.lr*(0-self.predict(weight,feat2[i]))*feat2[i][2]
                
                weight = n_weight
            
            count+=1
            c_rate = self.predict_all(weight,feat1,feat2)
            
            fp.write("{}. precent of correct answers : {}\n".format(count,c_rate))
            
            if c_rate > b_rate:
                b_rate = c_rate
                b_weight = weight
        
            if count == 1000:
                break
        print "Training ends!\nBest weight's number of correct answers = {}".format(b_rate)
        fp.close()

        # Return best weight
        return b_weight

# Get a argument in terminal
def runExp(lr):
    
    myfish = perceptron(lr)
    
    
    salmon_train,seabass_train = LoadFish("salmon_train.txt","seabass_train.txt")
    weight = myfish.train(salmon_train,seabass_train)

    # Predict test data with training weight
    fp = open("test_log_[{}].txt".format(lr),'w')
    salmon_test,seabass_test = LoadFish("salmon_test.txt","seabass_test.txt")
    print "Test result file : test_log_[{}].txt".format(lr)
    
    # Return correct count
    correct = myfish.predict_all(weight,salmon_test,seabass_test)
    fp.write("Weight : {}\nNumber of correct answers for test data : {}\n".format(weight,correct))

    fp.close()

    print "\nThank you\n"


if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 2:   # Check command line argument count
        learning_rate = float(sys.argv[1])
        
        runExp(learning_rate)
    else:
        print ('Usage : %s [learning rate]'
               )%(op.basename(sys.argv[0]))




