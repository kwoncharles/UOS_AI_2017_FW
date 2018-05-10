# -*- coding: utf-8 -*-

import os
import os.path as op
import struct as st
import numpy as np
import numpy.random as nr
import pickle as pkl

# test Data만 Load


# MNIST 데이터 경로
#_SRC_PATH = u'..\\'
#_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
#_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'

_TEST_DATA_FILE = 't10k-images.idx3-ubyte'
_TEST_LABEL_FILE = 't10k-labels.idx1-ubyte'

# MNIST 데이터 크기 (28x28)
_N_ROW = 28                 # 세로 28픽셀
_N_COL = 28                 # 가로 28픽셀
_N_PIXEL = _N_ROW * _N_COL


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
    # 테스트 데이터 / 레이블 로드
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    tsDataList, tsLabelList = loadMNIST()
    
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    


# weight를 pkl파일로 저장하는 method
def load(fn):
    fd = open(fn,'rb')
    obj = pkl.load(fd)
    fd.close()
    return obj


# Test Data reshape

for i in range(len(tsDataList)):
    tsDataList[i]=tsDataList[i].reshape(1,784)
    
# Test Label reshape as Onehot

tslabel_onehot=np.zeros([len(tsLabelList),10])

for i in range(0,len(tsLabelList)):
    tslabel_onehot[i][tsLabelList[i]] = 1
    

# Data사용이 용이하게 numpy array로 바꿔준다

tsDataList = np.array(tsDataList)

# Predict method
def predict(feat,label,weight):
    fp = open("test_output.txt",'wt')
    correct=0
        
    # 데이터를 순서대로 돌며 예측값과 정답을 비교
    for i in range(len(feat)):
        pred=np.argmax(np.dot(feat[i],weight))
        if(label[i][pred]==1.0):
            correct+=1
    avg=np.float32(correct)/len(feat)
    fp.write("Error rate of Test data : {}".format(1.0-avg))
    print "Error rate of Test data : {}".format(1.0-avg)
    return (1.0-avg)
        
# pkl 파일형태의 parameter를 불러온다        
loadedParam = load('best_param.pkl')
predict(tsDataList,tslabel_onehot,loadedParam)
