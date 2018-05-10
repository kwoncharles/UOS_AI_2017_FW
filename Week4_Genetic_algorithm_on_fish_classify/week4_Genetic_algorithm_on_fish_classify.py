#_*_ coding:utf-8 _*_

import numpy as np
import sys
import os.path as op
import random

# 물고기 파일 오픈 method
def OpenFish(file1,file2):
    
    fp = open(file1,"r")
    lines = fp.readlines()

    fp = open(file2,"r")
    lines2 = fp.readlines()

    fp.close()

    salmon = list() # 고기들의 정보를 저장할 list 선언
    seabass = list()

    for line in lines:
        salmon.append(((line.split()))) # split을 통해 문자열 내의 공백을 제외한 값을 list에 저장

    for line in lines2:
        seabass.append(((line.split())))

    # 문자열로 저장돼있는 값들을 float형으로 변환
    for i in range(len(salmon)):
        salmon[i] = map(float,salmon[i])

    for i in range(len(seabass)):
        seabass[i] = map(float,seabass[i])
    
    return salmon,seabass



# training data 파일 오픈 및 파일 내용 저장
salmon,seabass = OpenFish("salmon_train.txt","seabass_train.txt")

# test data 오픈 및 저장
t_salmon,t_seabass = OpenFish("salmon_train.txt","seabass_train.txt")



# fitness function
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
        
        # 가중치,적합도,최솟값 등을 저장할 변수 선언
        self.weight = []
        self.fit = []
        self.min_cost = 1000000
        self.min_weight = [0,0,0]
        self.cor_max = 0
        self.count = 0 # 학습횟수 카운팅 변수
        
        # train, test 파일 명 지정
        trResFn = 'train_log_%d_%d_%.2f.txt'%(self.entity,self.elite,self.mut_prob)
        self.fp = open(trResFn,'w')
        print 'Training result file:',trResFn
        
        self.tsResFn = 'test_output_%d_%d_%.2f.txt'%(self.entity,self.elite,self.mut_prob)
        self.fpT = open(self.tsResFn,'w')
        
        # 개체 수 만큼 초기 가중치 할당
        for i in range(self.entity):
            self.weight.append([random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),
                  random.uniform(-150,150)])
    


    # Fitness를 반환하는 함수
    def newfit(self,sal,seab,weight):

        # 연어들에 대한 cost 계산
        # unistep이 1이면 연어 0이면 농어
        cost_list = []
        for i in range(len(sal)):
            cost_list.append(abs((unitStep(func(sal[i],weight))-1))
                         *abs(func(sal[i],weight)))
        sum_cost = np.sum(cost_list)
        cost1 = (sum_cost/len(sal))

        # 농어들에 대한 cost 계산
        cost_list = []
        for i in range(len(seab)):
            cost_list.append(abs((unitStep(func(seab[i],weight))-0))
                             *abs(func(seab[i],weight)))
        sum_cost = np.sum(cost_list)
        cost2 = (sum_cost/len(seab))


        # 성능이 좋은 weight에게 큰 점수(fitness)를 주기 위해 cost에 역수를 취한다
        return 1/(cost1+cost2)
    
    # 정답예측
    def predict(self,sal,seab,weight):
        correct1 = 0
        correct2 = 0

        # 현재 가중치 값을 가지고 연어와 농어를 각각 분류하고 틀린 횟수를 각각 저장

        for i in range(len(sal)):
            if unitStep(func(sal[i],weight)) == 1:
                correct1 += 1


        for i in range(len(seab)):

            if unitStep(func(seab[i],weight)) == 0:
                correct2 += 1
        
        correct = (correct1 + correct2)
            
        self.fp.write("   Correct counts : {}\n\n".format(correct))
        
        # 전체 데이터 중 맞춘 개수 반환
        
        return correct
    
    # 테스트 예측용
    def Test_predict(self,sal,seab,weight):
        correct1 = 0
        correct2 = 0

        # 현재 가중치 값을 가지고 연어와 농어를 각각 분류하고 틀린 횟수를 각각 저장

        for i in range(len(sal)):
            if unitStep(func(sal[i],weight)) == 1:
                correct1 += 1


        for i in range(len(seab)):

            if unitStep(func(seab[i],weight)) == 0:
                correct2 += 1
        
        correct = (correct1 + correct2)
        
        self.fpT.write("===== Test result =====\n")
        self.fpT.write("The number of correct answers : {} / {}\n\n".format(correct,len(sal)+len(seab)))
        
    
    # 확률적으로 돌연변이를 생성 (call by ref.)
    def make_mut(self,prob,weight):
        num = random.uniform(0,1)
        # 랜덤한 숫자가 prob보다 작을 경우 돌연변이로 만듦
        if(num < prob):
            weight[0] += random.uniform(-0.1,0.1)
            weight[1] += random.uniform(-0.1,0.1)
            weight[2] += random.uniform(-50,50)
            self.fp.write("    Mutant was born! X_X\n\n")

    # Training
    def train(self):
        self.count += 1 # 학습 횟수를 증가시킨다
        self.fp.write("Train {}.\n".format(self.count))
        
        
        # 각 유전자들의 fitness 계산
        self.fit = []
        for i in range(self.entity):
            self.fit.append(self.newfit(salmon,seabass,self.weight[i]))

        # 유전자들을 fitness가 높은 순서대로 정렬
        sortidx = np.argsort(self.fit)[::-1]
        fit_sum=np.sum(self.fit)
        fit_prob =[]
        
        # fitness가 높은 유전자들에게 높은 백분율 값을 부여
        for i in range(self.entity):
            c_fit = self.fit[i]
            prob= ((c_fit/fit_sum)*100)
            fit_prob.append(int(prob))
            
        nweight = []
        # elite 유전자를 새로운 weight 리스트에 먼저 삽입
        for i in range(self.elite):
            nweight.append(self.weight[sortidx[i]])
            
        # candidate 리스트에는 상위 순위의 weight들을 백분율만큼 저장
        # candidate 리스트에는 100개의 weight가 들어있음
        # self.weight[3] 의 cost가 40%의 가중치를 갖는다면 
        # candidate리스트에는 self.weight[3]가 40개 들어있다
        
        candi=[]
        for i in range(self.entity):
            for j in range(fit_prob[i]):
                candi.append(self.weight[i])       
        
        # 새롭게 태어날 개체들을 
        # candidate배열에 있는 개체들 중에 골라 랜덤으로 채운다
        for i in range(self.entity-self.elite):
            a=int(random.uniform(0,len(candi)))
            b=int(random.uniform(0,len(candi)))
            c=int(random.uniform(0,len(candi)))
            n=[candi[a][0],candi[b][1],candi[c][2]]
            # 매 루프마다 뮤턴트 함수를 호출
            # 확률에 따라 뮤턴트가 될 수도 안될 수도 있음
            self.make_mut(self.mut_prob,n)
            nweight.append(n)
        
        # fitness에 역수를 취해줘서 best 유전자의 cost 출력
        c_min_cost = 1/self.fit[sortidx[0]]
        
        self.fp.write("   Current cost : %.6f\n"%(c_min_cost))
        
        
        # minimum cost 저장
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
