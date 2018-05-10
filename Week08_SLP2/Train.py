# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import numpy.random as nr
import pickle as pkl

# MNIST 데이터 경로
#_SRC_PATH = u'..\\'
#_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
#_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
#_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
#_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

_TRAIN_DATA_FILE = 'train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = 'train-labels.idx1-ubyte'
_TEST_DATA_FILE = 't10k-images.idx3-ubyte'
_TEST_LABEL_FILE = 't10k-labels.idx1-ubyte'


# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL



def drawImage(dataArr, fn):
    fig, ax = plt.subplots()
    ax.imshow(dataArr, cmap='gray')
    plt.show()
    plt.savefig(fn)
    
    
    
def loadData(fn):
    print 'loadData', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    nRow = st.unpack('>I', fd.read(4))[0]
    nCol = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    print 'nRow', nRow
    print 'nCol', nCol
    
    # data: unsigned byte
    dataList = []
    for i in range(nData):
        dataRawList = fd.read(_N_PIXEL)
        dataNumList = st.unpack('B' * _N_PIXEL, dataRawList)
        #dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        dataArr = np.array(dataNumList)
        # overflow 수정
        dataList.append(dataArr.astype('float32')/255.0)
        
    fd.close()
    
    print 'done.'
    print
    
    return dataList
    


def loadLabel(fn):
    print 'loadLabel', fn
    
    fd = open(fn, 'rb')
    
    # header: 32bit integer (big-endian)
    magicNumber = st.unpack('>I', fd.read(4))[0]
    nData = st.unpack('>I', fd.read(4))[0]
    
    print 'magicNumber', magicNumber
    print 'nData', nData
    
    # data: unsigned byte
    labelList = []
    for i in range(nData):
        dataLabel = st.unpack('B', fd.read(1))[0]
        labelList.append(dataLabel)
        
    fd.close()
    
    print 'done.'
    print
    
    return labelList



def loadMNIST():
    # 학습 데이터 / 레이블 로드
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return trDataList, trLabelList, tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    trDataList, trLabelList, tsDataList, tsLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    


# weight를 pkl파일로 저장하는 method
def save(fn, obj):
    fd = open(fn,'wb')
    pkl.dump(obj,fd)
    fd.close()


# Training, Test Data reshape

for i in range(len(trDataList)):
    trDataList[i]=trDataList[i].reshape(1,784)
for i in range(len(tsDataList)):
    tsDataList[i]=tsDataList[i].reshape(1,784)
    
# Training, Test Label reshape as Onehot

trlabel_onehot=np.zeros([len(trLabelList),10])

for i in range(0,len(trLabelList)):
    trlabel_onehot[i][trLabelList[i]] = 1
    
tslabel_onehot=np.zeros([len(tsLabelList),10])

for i in range(0,len(tsLabelList)):
    tslabel_onehot[i][tsLabelList[i]] = 1
    

# Data사용이 용이하게 numpy array로 바꿔준다
    
trDataList = np.array(trDataList)
tsDataList = np.array(tsDataList)

# Single layer perceptron
class SLP:
    # class선언 시 learning rate를 인자로 받는다
    def __init__(self,lr):
        self.lr = lr
        self.weight = nr.rand(784,10)
        # Batch size, Training epoch 횟수 설정
        self.batch_size = 10000
        self.tr_epochs = 10
        
    def activation(self,x):
        return (1/(1+np.exp(-x)))
    
    def train(self,feat,label):
        
        total_batch = len(feat)/self.batch_size
        
        # 분류 결과가 가장 좋은 파라미터를 저장하기 위해 변수 선언
        self.best_weight=self.weight
        self.best_error=100.0
        fp = open("train_log.txt",'wt')
        
        print "\nLearning starts!"
        for epoch in range(self.tr_epochs):
            
            # 매 학습마다 데이터를 섞어주기 위해 Index변수 따로 선언
            trainIdx = range(0,len(feat))
            nr.shuffle(trainIdx)
            
            fp.write("\n--------- epoch{} ---------\n".format(epoch+1))
            print "\n--------- epoch{} ---------".format(epoch+1)
            for i in range(total_batch):
                cost = 0
                
                for j in range(i*self.batch_size,(i+1)*self.batch_size):
                    # 갱신할 weight를 임시로 저장할 변수 선언
                    new_weight=self.weight
                    
                    # 활성화함수 통과
                    result=self.activation(np.dot(feat[trainIdx[j]]/10,self.weight))
                    
                    # weight 갱신 (column별로 갱신)
                    for k in range(10):
                        # weight 갱신을 위해 reshape을 해준다
                        new_weight[:,[k]] = new_weight[:,[k]] + (self.lr*(label[trainIdx[j]][k]-result[0][k]) \
                                    *result[0][k]*(1-result[0][k])*feat[trainIdx[j]]).reshape([784,1])
                    ## 여기 잘못된듯?
                    
                    # 평균 cost 계산
                    cost += (np.sum(label[trainIdx[j]]-result)**2)/2
                    
                    self.weight=new_weight
                    
                # 평균 cost 계산
                cost = cost/self.batch_size
                fp.write("{}. cost: {}\n".format(j+1,np.sum(cost)))
                print "{}. cost: {}".format(j+1,np.sum(cost))
                
                error=self.predict(feat,label,self.weight)
                fp.write("Error_rate: {}\n".format(error))
                print("Error_rate: {}".format(error))
                
                # 분류 결과가 지금까지 중 가장 좋았다면 결과와 weight을 저장
                if(error<self.best_error):
                    self.best_error=error
                    self.best_weight=self.weight
                    
                # learning rate decay
                self.lr = self.lr*0.99
        print "\nLearning ends!\n"
                    
        
    def predict(self,feat,label,weight):
        correct=0
        
        # 데이터를 순서대로 돌며 예측값과 정답을 비교
        for i in range(len(feat)):
            pred=np.argmax(np.dot(feat[i],weight))
            if(label[i][pred]==1.0):
                correct+=1
        avg=np.float32(correct)/len(feat)
        return (1.0-avg)
        
        
test = SLP(0.05)
test.train(trDataList,trlabel_onehot)

# 학습 결과가 가장 좋은 weight pkl파일로 저장
param = test.best_weight
save('best_param.pkl',param)
