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
import matplotlib.pyplot as plt

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

def drawFeatMap(dstFn, featMap):
    fig, ax = plt.subplots()
    ax.imshow(featMap, cmap='gray')
    plt.savefig(dstFn)
    
    #flush
    plt.cla()
    plt.clf()
    plt.close()



if __name__ == '__main__':
    tsDataList, tsLabelList = loadMNIST()
    
    print 'len(tsDataList)', len(tsDataList)
    print 'len(tsLabelList)', len(tsLabelList)


# Test Label reshape as Onehot
tslabel_onehot=ku.np_utils.to_categorical(tsLabelList,num_classes=10)

# Save as numpy array
tsDataList = np.array(tsDataList).reshape(10000,28,28,1)
tsLabelList = np.array(tsLabelList)

# Import model of the first convolution layer
modelconv = km.load_model('modelconv1.h5')

res = modelconv.predict(tsDataList, batch_size=4)

# class 9 : [7,9,12]
# class 2 : [1,35,38]

# Save the Feature maps in the first Convolution layer
d
# Feature of class '2'  // Index 1, 35, 38
for i in range(3):
    for j in range(3):
        drawFeatMap("[C2]_[S%d]_[F%d]"%(i+1,j+1), res[[1,35,38][i],:,:,j])


# Feature of class '9'  // Index 7, 9, 12
for i in range(3):
    for j in range(3):
        drawFeatMap("[C9]_[S%d]_[F%d]"%(i+1,j+1),res[[7,9,12][i],:,:,j])

# Import Full model
model = km.load_model('modelFull.h5')

res = model.predict(tsDataList, batch_size=4)
res = np.argmax(res, axis=1)

correct=0
for i in range(len(tsLabelList)):
    if(tsLabelList[i] == res[i]):
        correct+=1

print "%d/%d"%(correct,len(tsLabelList))

fp = open('test_output','wt')
fp.write("Test result\n")
fp.write("\nCorrect answers : %d/%d\nError rate : %.3f"%(correct,len(tsLabelList),1.0-(np.float(correct)/len(tsLabelList))))
fp.close()
