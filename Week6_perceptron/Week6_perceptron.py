# _*_ coding: utf-8 _*_

import numpy as np
import random
import sys
import os.path as op

# 물고기 data를 불러오기 위한 LoadFish 정의
def LoadFish(file1, file2):
    fp = open(file1,'r')
    lines = fp.readlines()
    
    fp = open(file2,'r')
    lines2 = fp.readlines()
    
    fp.close()
    
    salmon = []
    seabass = []
    
    # 물고기들 data를 공백을 기준으로 나누어 리스트에 저장
    for line in lines:
        salmon.append(line.split())
    for line in lines2:
        seabass.append(line.split())
    
    # 데이터타입을 float형으로 모두 바꿔준다
    for i in range(len(salmon)):
        salmon[i] = map(float,salmon[i])
    for i in range(len(seabass)):
        seabass[i] = map(float,seabass[i])
    
    # bias에 곱해질 1.0을 각 데이터의 idx0번 자리에 넣어준다
    for i in range(len(salmon)):
        salmon[i] = [1.0,salmon[i][0],salmon[i][1]]
    for i in range(len(seabass)):
        seabass[i] = [1.0,seabass[i][0],seabass[i][1]]
    
    return salmon,seabass

# 어종 분류를 위한 퍼셉트론 클래스
class perceptron:
    
    #learning rate를 인자로 받는다
    def __init__(self,lr):
        self.lr = lr
    
    # w*x + b 값이 0보다 크면 연어, 작거나 같으면 농어로 분류한다
    def predict(self,weight,feat):
        if(weight[0]*feat[0] + weight[1]*feat[1] + weight[2]*feat[2] > 0):
            return 1 # salmon
        else:
            return 0 # seabass

    # 인자로 받은 weight값을 가지고 어종을 분류한다.
    # 정답을 맞춘 개수를 반환한다.
    def predict_all(self,weight,salmon,seabass):
        correct = 0
        
        for i in range(len(salmon)):
            if(self.predict(weight,salmon[i]) == 1):
                correct+=1
        for i in range(len(seabass)):
            if(self.predict(weight,seabass[i]) == 0):
                correct+=1
        
        return correct # 정답개수 반환


    # 학습 method
    def train(self,feat1,feat2):
        # 초기 가중치를 랜덤으로 할당
        weight = [random.uniform(-20,20),random.uniform(-20,20),random.uniform(0,50)]
        
        # 매 루프마다 현재 가중치를 가지고 predict_all 메서드 호출
        # 가장 좋은 결과가 나올 때마다 best rate와 best weight을 갱신시킨다
        b_weight = [0,0,0]
        b_rate = 0
        
        count = 0
        
        fp=open("train_log_[{}].txt".format(self.lr),'w')
        print "Training starts!\nTraining result file : train_log_[{}].txt\n".format(self.lr)
        print "Initial weight's number of correct answer : {}\n".format(self.predict_all(weight,feat1,feat2))

        while(True):
            for i in range(len(feat1)):
                # 갱신된 가중치를 임시로 저장할 n_weight 리스트 초기화
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
            
            # 현재 정답률(c_rate)가 이전 최고 정답률(b_rate)보다 크면
            # b_rate과 b_weight을 갱신시킨다
            if c_rate > b_rate:
                b_rate = c_rate
                b_weight = weight
            
            # 학습이 1000번 진행될 시 loop 탈출
            if count == 1000:
                break
        print "Training ends!\nBest weight's number of correct answers = {}".format(b_rate)
        fp.close()

        # 학습결과가 가장 좋았던 weight를 반환
        return b_weight


# Terminal에서 argument를 입력받아 학습 진행
def runExp(lr):
    
    # myfish라는 퍼셉트론을 정의
    myfish = perceptron(lr)
    
    
    salmon_train,seabass_train = LoadFish("salmon_train.txt","seabass_train.txt")
    weight = myfish.train(salmon_train,seabass_train)

    # training weight으로 test data에 대한 predict 진행
    fp = open("test_log_[{}].txt".format(lr),'w')
    salmon_test,seabass_test = LoadFish("salmon_test.txt","seabass_test.txt")
    print "Test result file : test_log_[{}].txt".format(lr)
    
    # 맞춘 개수를 반환받는다
    correct = myfish.predict_all(weight,salmon_test,seabass_test)
    fp.write("Weight : {}\nNumber of correct answers for test data : {}\n".format(weight,correct))

    fp.close()

    print "\nThank you\n"


if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 2:   # command line argument 수 확인
        learning_rate = float(sys.argv[1])
        
        runExp(learning_rate)
    else:
        print ('Usage : %s [learning rate]'
               )%(op.basename(sys.argv[0]))




