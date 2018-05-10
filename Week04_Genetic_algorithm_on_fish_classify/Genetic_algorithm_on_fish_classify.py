#_*_ coding:utf-8 _*_

import numpy as np
import sys
import os.path as op
import random

def OpenFish(file1,file2):
    
    fp = open(file1,"r")
    lines = fp.readlines()

    fp = open(file2,"r")
    lines2 = fp.readlines()

    fp.close()

    salmon = list()
    seabass = list()

    for line in lines:
        salmon.append(((line.split())))

    for line in lines2:
        seabass.append(((line.split())))

    # Convert string into float
    for i in range(len(salmon)):
        salmon[i] = map(float,salmon[i])

    for i in range(len(seabass)):
        seabass[i] = map(float,seabass[i])
    
    return salmon,seabass



# Training data
salmon,seabass = OpenFish("salmon_train.txt","seabass_train.txt")

# Test data
t_salmon,t_seabass = OpenFish("salmon_train.txt","seabass_train.txt")



# Fitness function
def func(x,param):
    newp = x[0]*param[0]+x[1]*param[1]+param[2]
    return x[0]+x[1]-newp
    
def unitStep(x):
    if x<0:
        return 0
    return 1

# Genetic algorithm class
class Genetic:
    def __init__(self,entity,elite,mut_prob):
        self.entity = entity
        self.elite = elite
        self.mut_prob = mut_prob
        
        self.weight = []
        self.fit = []
        self.min_cost = 1000000
        self.min_weight = [0,0,0]
        self.cor_max = 0
        self.count = 0 # Training count
        
        trResFn = 'train_log_%d_%d_%.2f.txt'%(self.entity,self.elite,self.mut_prob)
        self.fp = open(trResFn,'w')
        print 'Training result file:',trResFn
        
        self.tsResFn = 'test_output_%d_%d_%.2f.txt'%(self.entity,self.elite,self.mut_prob)
        self.fpT = open(self.tsResFn,'w')
        
        # Allocate initial weights as much as its numbers
        for i in range(self.entity):
            self.weight.append([random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),
                  random.uniform(-150,150)])
    


    # Return new Fitness
    def newfit(self,sal,seab,weight):

        # Compute salmon's cost
        # If unistep is 1 then salmon, else seabass
        cost_list = []
        for i in range(len(sal)):
            cost_list.append(abs((unitStep(func(sal[i],weight))-1))
                         *abs(func(sal[i],weight)))
        sum_cost = np.sum(cost_list)
        cost1 = (sum_cost/len(sal))

        # Compute seabass's cost
        cost_list = []
        for i in range(len(seab)):
            cost_list.append(abs((unitStep(func(seab[i],weight))-0))
                             *abs(func(seab[i],weight)))
        sum_cost = np.sum(cost_list)
        cost2 = (sum_cost/len(seab))


        # Make a reciprocal to give a high score to good weight
        return 1/(cost1+cost2)
    
    # Predict
    def predict(self,sal,seab,weight):
        correct1 = 0
        correct2 = 0

        # Count wrong answers with current weight
        for i in range(len(sal)):
            if unitStep(func(sal[i],weight)) == 1:
                correct1 += 1


        for i in range(len(seab)):

            if unitStep(func(seab[i],weight)) == 0:
                correct2 += 1
        
        correct = (correct1 + correct2)
            
        self.fp.write("   Correct counts : {}\n\n".format(correct))
        
        # Return correct answers' count
        return correct
    
    # Predict_test_data
    def Test_predict(self,sal,seab,weight):
        correct1 = 0
        correct2 = 0

        # Classify with current weight and save the wrong answers' count
        for i in range(len(sal)):
            if unitStep(func(sal[i],weight)) == 1:
                correct1 += 1


        for i in range(len(seab)):

            if unitStep(func(seab[i],weight)) == 0:
                correct2 += 1
        
        correct = (correct1 + correct2)
        
        self.fpT.write("===== Test result =====\n")
        self.fpT.write("The number of correct answers : {} / {}\n\n".format(correct,len(sal)+len(seab)))
        
    # Make a mutant stochastically(call by ref.)
    def make_mut(self,prob,weight):
        num = random.uniform(0,1)
        # if random number is lower than prob, make a mutant
        if(num < prob):
            weight[0] += random.uniform(-0.1,0.1)
            weight[1] += random.uniform(-0.1,0.1)
            weight[2] += random.uniform(-50,50)
            self.fp.write("    Mutant was born! X_X\n\n")

    # Training
    def train(self):
        self.count += 1
        self.fp.write("Train {}.\n".format(self.count))
        
        # Compute each gene's fitness
        self.fit = []
        for i in range(self.entity):
            self.fit.append(self.newfit(salmon,seabass,self.weight[i]))

        # Sort gene's with its fitness
        sortidx = np.argsort(self.fit)[::-1]
        fit_sum=np.sum(self.fit)
        fit_prob =[]
        
        # Give a high percentage to high fitness genes
        for i in range(self.entity):
            c_fit = self.fit[i]
            prob= ((c_fit/fit_sum)*100)
            fit_prob.append(int(prob))
            
        nweight = []
        
        # Insert first the elite genes
        for i in range(self.elite):
            nweight.append(self.weight[sortidx[i]])
            
        # Save the top class weights in candi list as much as their percentage
        # There are list of 100 in candi list
        # If the cost of self.weight[3] has a weight with 40%
        # There are 40 self.weight[3] in candi list
        
        candi=[]
        for i in range(self.entity):
            for j in range(fit_prob[i]):
                candi.append(self.weight[i])       

        # Select the newborn gene in candi randomly
        for i in range(self.entity-self.elite):
            a=int(random.uniform(0,len(candi)))
            b=int(random.uniform(0,len(candi)))
            c=int(random.uniform(0,len(candi)))
            n=[candi[a][0],candi[b][1],candi[c][2]]
            
            # Call mutant function in every loop
            # It also cannot be a mutant (stochastically)
            self.make_mut(self.mut_prob,n)
            nweight.append(n)

        # Print the cost of best gene
        c_min_cost = 1/self.fit[sortidx[0]]
        
        self.fp.write("   Current cost : %.6f\n"%(c_min_cost))
        
        
        # Save a minimum cost
        cor=self.predict(salmon,seabass,self.weight[sortidx[0]])
        if ( c_min_cost < self.min_cost and cor > self.cor_max ):
            self.min_cost = c_min_cost
            self.min_weight = self.weight[sortidx[0]]
            self.cor_max = cor
    
        
        self.fp.write("   Best gene's cost : %.6f\n"%(self.min_cost))    
        self.fp.write("   Best gene's number of correct answers so far : {}\n\n".format(self.cor_max))
        self.weight = nweight

        


# Terminal에서 argument를 입력받아 학습 진행
def runExp(popSize, eliteNum, mutProb):

    
    print '\nTraining starts!'
    
    gene = Genetic(popSize,eliteNum,mutProb)
    
    for i in range(3000):
        gene.train()

    print 'training ends!'

    gene.fp.write('\n\ntraining ends!\n\n\n')

    gene.fp.write ('\nBest result\'s cost : %.6f'%(gene.min_cost))
    gene.fp.write ('\nIts weight : {}'.format(gene.min_weight))
    gene.fp.write ('\nBest number of correct answers : {}'.format(gene.cor_max))

    gene.fp.close()
    
    # training weight으로 test data에 대한 predict 진행
    gene.Test_predict(t_salmon,t_seabass,gene.min_weight)
    print 'Test result file:',gene.tsResFn
    gene.fpT.close()

if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 4:   # command line argument 수 확인
        popSize = int(sys.argv[1])
        eliteNum = int(sys.argv[2])
        mutProb = float(sys.argv[3])
        
        runExp(popSize, eliteNum, mutProb)
    else:
        print ('Usage : %s [populationSize] [eliteNum]''[mutationProb]'
               )%(op.basename(sys.argv[0]))
