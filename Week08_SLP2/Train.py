# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import numpy.random as nr
import pickle as pkl

# MNIST data route
_SRC_PATH = u'..\\'
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

#_TRAIN_DATA_FILE = 'train-images.idx3-ubyte'
#_TRAIN_LABEL_FILE = 'train-labels.idx1-ubyte'
#_TEST_DATA_FILE = 't10k-images.idx3-ubyte'
#_TEST_LABEL_FILE = 't10k-labels.idx1-ubyte'


# MNIST data size(28x28)
_N_ROW = 28
_N_COL = 28
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
        # overflow
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
    # Load training data/label
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    # Load test data/label
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return trDataList, trLabelList, tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    trDataList, trLabelList, tsDataList, tsLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    

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
    

# Save as numpy array
    
trDataList = np.array(trDataList)
tsDataList = np.array(tsDataList)

# Single layer perceptron
class SLP:
    # An argument is learning rate
    def __init__(self,lr):
        self.lr = lr
        self.weight = nr.rand(784,10)
        # Batch size, Training epoch count
        self.batch_size = 10000
        self.tr_epochs = 10
        
    def activation(self,x):
        return (1/(1+np.exp(-x)))
    
    def train(self,feat,label):
        
        total_batch = len(feat)/self.batch_size
        
        self.best_weight=self.weight
        self.best_error=100.0
        fp = open("train_log.txt",'wt')
        
        print "\nLearning starts!"
        for epoch in range(self.tr_epochs):
            
            # Shuffling Idx
            trainIdx = range(0,len(feat))
            nr.shuffle(trainIdx)
            
            fp.write("\n--------- epoch{} ---------\n".format(epoch+1))
            print "\n--------- epoch{} ---------".format(epoch+1)
            for i in range(total_batch):
                cost = 0
                
                for j in range(i*self.batch_size,(i+1)*self.batch_size):
                    new_weight=self.weight
                    
                    # Activation function
                    result=self.activation(np.dot(feat[trainIdx[j]]/10,self.weight))
                    
                    # Renew weight (by column)
                    for k in range(10):
                        # Reshape for renewal
                        new_weight[:,[k]] = new_weight[:,[k]] + (self.lr*(label[trainIdx[j]][k]-result[0][k]) \
                                    *result[0][k]*(1-result[0][k])*feat[trainIdx[j]]).reshape([784,1])

                    # Compute mean cost
                    cost += (np.sum(label[trainIdx[j]]-result)**2)/2
                    
                    self.weight=new_weight
                    
                # Compute mean cost
                cost = cost/self.batch_size
                fp.write("{}. cost: {}\n".format(j+1,np.sum(cost)))
                print "{}. cost: {}".format(j+1,np.sum(cost))
                
                error=self.predict(feat,label,self.weight)
                fp.write("Error_rate: {}\n".format(error))
                print("Error_rate: {}".format(error))
                
                if(error<self.best_error):
                    self.best_error=error
                    self.best_weight=self.weight
                    
                # learning rate decay
                self.lr = self.lr*0.99
        print "\nLearning ends!\n"
                    
        
    def predict(self,feat,label,weight):
        correct=0
        
        # Compare predict and correct value
        for i in range(len(feat)):
            pred=np.argmax(np.dot(feat[i],weight))
            if(label[i][pred]==1.0):
                correct+=1
        avg=np.float32(correct)/len(feat)
        return (1.0-avg)
        
        
test = SLP(0.05)
test.train(trDataList,trlabel_onehot)

# Save best parameters
param = test.best_weight
save('best_param.pkl',param)
