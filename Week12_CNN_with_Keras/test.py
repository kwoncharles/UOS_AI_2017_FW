# -*- coding: utf-8 -*-
# Test_file

import os
import struct as st
import numpy as np
import numpy.random as nr
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku

nr.seed(12345)  # random seed



# MNIST data route
_SRC_PATH = u'..\\'
_TEST_DATA_FILE = _SRC_PATH + u'\\t10k-images.idx3-ubyte'
_TEST_LABEL_FILE = _SRC_PATH + u'\\t10k-labels.idx1-ubyte'
#_TEST_DATA_FILE = 't10k-images.idx3-ubyte'
#_TEST_LABEL_FILE = 't10k-labels.idx1-ubyte'


# MNIST data size (28x28)
_N_ROW = 28
_N_COL = 28
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
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        #dataArr = np.array(dataNumList)
        # overflow
        dataList.append(dataArr.astype('float32'))
        
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
    # Load Test data / label
    tsDataList = loadData(_TEST_DATA_FILE)
    tsLabelList = loadLabel(_TEST_LABEL_FILE)
    
    return tsDataList, tsLabelList

    
    
if __name__ == '__main__':
    tsDataList, tsLabelList = loadMNIST()
    
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)
    
    
# Test Label reshape as Onehot
tslabel_onehot=ku.np_utils.to_categorical(tsLabelList,num_classes=10)

# Save as numpy array
tsDataList = np.array(tsDataList).reshape(10000,28,28,1)

# Load model that has best parameter
model = km.load_model('best_param.h5')

res = model.predict(tsDataList, batch_size=4)
res= np.argmax(res, axis=1)

correct=0
for i in range(len(tsLabelList)):
    if(tsLabelList[i] == res[i]):
        correct+=1

print "%d/%d"%(correct,len(tsLabelList))

fp = open('test_output.txt','wt')
fp.write("Test result\n")
fp.write("\nCorrect answers : %d/%d\nError rate : %.4f"%(correct,len(tsLabelList),1.0-(np.float(correct)/len(tsLabelList))))
fp.close()
