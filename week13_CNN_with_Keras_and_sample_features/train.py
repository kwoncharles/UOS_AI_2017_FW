# -*- coding: utf-8 -*-
# Training_file

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
_TRAIN_DATA_FILE = _SRC_PATH + u'\\train-images.idx3-ubyte'
_TRAIN_LABEL_FILE = _SRC_PATH + u'\\train-labels.idx1-ubyte'

#_TRAIN_DATA_FILE = 'train-images.idx3-ubyte'
#_TRAIN_LABEL_FILE = 'train-labels.idx1-ubyte'


# MNIST date size (28x28)
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
        dataArr = np.array(dataNumList).reshape(_N_ROW, _N_COL)
        #dataArr = np.array(dataNumList)
        # overflow
        dataList.append(dataArr.astype('float32')/255.0)
    
    fd.close()
    
    print 'done.'
    
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
    # Load Training data / label
    trDataList = loadData(_TRAIN_DATA_FILE)
    trLabelList = loadLabel(_TRAIN_LABEL_FILE)
    
    return trDataList, trLabelList



if __name__ == '__main__':
    trDataList, trLabelList = loadMNIST()
    
    print 'len(trDataList)', len(trDataList)
    print 'len(trLabelList)', len(trLabelList)

# Test Label reshape as Onehot
trlabel_onehot=ku.np_utils.to_categorical(trLabelList,num_classes=10)

# Save as numpy array
trDataList = np.array(trDataList).reshape(60000,28,28,1)

# Compose model
inputFeat = kl.Input(shape=(28, 28, 1))

conv1 = kl.Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same')(inputFeat)

relu1 = kl.Activation('relu')(conv1)
drop1 = kl.Dropout(rate=0.1)(relu1)
conv2 = kl.Conv2D(filters=5, kernel_size=(5, 5), strides=2, padding='same')(drop1)
relu2 = kl.Activation('relu')(conv2)
drop2 = kl.Dropout(rate=0.1)(relu2)
flatten = kl.Flatten()(drop2)
dense = kl.Dense(units=10)(flatten)
output = kl.Activation('sigmoid')(dense)

modelFull = km.Model(inputs=[inputFeat], outputs = [output])
modelConv1 = km.Model(inputs=[inputFeat], outputs = [conv1])

modelFull.compile(loss='mean_squared_error',optimizer=ko.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

# Train model and save the log
fp = open('train_log','wt')

epoch_num = 3

train_log=modelFull.fit(trDataList, trlabel_onehot, epochs=epoch_num, batch_size=3,validation_split=0.1)

fp.write("Train log\n\n")

for i in range(epoch_num):
    fp.write("epoch%d loss : %.3f , accuracy : %.3f\n"%(i+1,train_log.history['loss'][i], train_log.history['acc'][i]))

fp.close()

modelFull.save('modelFull.h5')
modelConv1.save('modelConv1.h5')
