
# coding: utf-8

# In[604]:

import numpy as np
np.random.seed(123)

from keras.layers import Input, Dense
from keras.models import Model

import pandas
import os
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pylab as plt
######important
get_ipython().magic('matplotlib inline')
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
#from keras.optimizers import RMSprop


# In[685]:

#Test with the case of time over stock
#并区分训练集和测试集
def load_factor(file,train_perc):
    FacData = pandas.read_csv(file,engine='python', skipfooter=3)
    print(len(FacData))
    #dataframe = pandas.read_csv("inte.csv",usecols=[1], engine='python', skipfooter=3)
    fac_num=FacData.shape[1]-2
    print("Factor个数为："+str(fac_num))

    dataset=FacData.iloc[:,2:fac_num+2].values.astype('float32')  #为了可视化我取25个
    train_size=int(len(dataset)*train_perc)
    test_size=len(dataset)-train_size
#train, test = dat[0:train_size], dat[train_size:len(dataset)]
    x_train, x_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return FacData,x_train,x_test

def autoencoder_train(input_encoded,fun_en,fun_de,dim_en,dim_de):
    encoded = Dense(dim_en, activation=fun_en)(input_encoded)  #dim is output dimension
    decoded = Dense(dim_de, activation=fun_de)(encoded)
    autoencoder = Model(input=input_encoded, output=decoded)
    encoder=Model(input=input_encoded,output=encoded)
    #train our autoencoder for 100 epochs
    # expected input batch shape: (batch_size, timesteps, data_dim)
    #batch_size: subset size of your training sample going to be used in order to train the network
    #batch: Each batch trains network in a successive order, taking into account the updated weights coming from the appliance of the previous batch.
    #nb_epoch=100 #number of iterations on the data
    #configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy'])
    #autoencoder.compile(optimizer='sgd', loss='mean_squared_error') 
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(input=encoded, output=decoder_layer(encoded))
    return autoencoder,encoder

def restruct_picture(x_test,decoded_imgs,dim):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(dim, dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(dim,dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def split_date(day):
    date=day.split('-')
    date=date[0]+date[1]+date[2]

    return date

def load_retu(file,C_data,L_day):
    Ret=pandas.read_csv(file,engine='python', skipfooter=3)
    # Y分为两档，换仓期间收益率表现为前20%的为强势股标为1，后20%为弱势标为0
    Y_data=Ret.rename(columns={'tdate': 'Date','symbol':'Symbol'})
    #print(len(C_data))
    #print(Y_data.head())
    M=pandas.merge(C_data,Y_data,how='inner',on=['Date','Symbol'])
    #print(M.head())
    #print(len(M))
    
    #区分训练集和测试集,以最后一个换仓日为界
    lastday=split_date(L_day)
    C_train=M[M.Date<int(lastday)]
    C_test=M[M.Date==int(lastday)]
    #print(C_test)
    fac_num=C_data.shape[1]-2
    TrainX=np.array(C_train.iloc[:,2:fac_num+2].values.astype('float32'))
    TestX=np.array(C_test.iloc[:,2:fac_num+2].values.astype('float32')) 
    TrainY=C_train['label'].astype('category')
    TrainY=np.array(TrainY.tolist())
    TestY=C_test['label'].astype('category')
    TestY=np.array(TestY.tolist())
    rval=[(TrainX,TrainY),(TestX,TestY)]
    return rval
   
def classifier(lr, decay, momentum,dim,cl):
    #在autoencoder的最后一层接入softmax分类器，利用前三周的数据对分类器做训练
    model = Sequential()
    #model.add(Dense(input_dim=encoding_dim,output_dim=8,init='normal',activation='tanh'))

    model.add(Dense(input_dim=dim,output_dim=cl,init='uniform',activation='softmax')) #只看换仓日当天压缩后的factor
    sgd = SGD(lr, decay, momentum, nesterov=True)
    early_stopping=EarlyStopping(monitor='val_loss', patience=2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
# fit the network
    return model

def train_classifier(model,TrainX,Train_Y,TestX,Test_Y,nb):
    history = model.fit(TrainX, Train_Y, nb_epoch=nb, batch_size=100,shuffle=True, validation_split=0.2)
# evaluate the network
    loss, accuracy = model.evaluate(TestX, Test_Y,verbose=0)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
    return model,loss,accuracy


# In[653]:

if __name__=='__main__':  

#################load factor data, shuffled and split between tran and test sets##############3
    FacData,x_train,x_test=load_factor('train1.csv',train_perc=0.75)
    FacData2,x_train2,x_test2=load_factor('train2.csv',train_perc=0.75)
    FacData3,x_train3,x_test3=load_factor('train3.csv',train_perc=0.75)
    FacData=FacData.append(FacData2)
    FacData=FacData.append(FacData3)
    x_train=np.vstack((x_train,x_train2)) 
    x_train=np.vstack((x_train,x_train3))
    x_test=np.vstack((x_test,x_test2))
    x_test=np.vstack((x_test,x_test3))
    print(len(FacData))
    print(x_train.shape)
##################Pretraining：AutoEncoder###########################################
###########Structure of Autoencoder Network: 4 Layer########################################

    
    #dim=[len(x_train[0]),24,16,8] # LayerDim的成员是每一层神经网络的节点数量
    dim=[len(x_train[0]),16,8]
    #####################Framework of Stacked autoencoder:4 layer####################################
# "encoded" is the encoded representation of the features
# this is the size of our encoded representations
#dim的成员是每一层神经网络的节点数量
    encoding_dim = dim[-1]  # 32->24->32, 24->16->24, 16->8->16
    # this is our input placeholder
    print(encoding_dim)
    
####The first autoencoder
    input_img = Input(shape=(dim[0],))
    encoded1 = Input(shape=(dim[1],))
    encoded2 = Input(shape=(dim[2],))


# In[654]:

np.random.seed(10)
autoencoder1,encoder1=autoencoder_train(input_img,'relu','relu',dim[1],dim[0])#encoded is the placeholder
#print(encoder1)


# In[660]:

early_stopping=EarlyStopping(monitor='val_loss', patience=2)
autoencoder1.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=400,
                shuffle=True,
                validation_data=(x_test, x_test),callbacks=[early_stopping])


# In[661]:

autoencoder2,encoder2=autoencoder_train(encoded1,'relu','sigmoid',dim[2],dim[1])   #encoder is the predictor
encode1_train=encoder1.predict(x_train)
encode1_test=encoder1.predict(x_test)
print(encode1_train)


# In[663]:

autoencoder2.fit(encode1_train, encode1_train,
                nb_epoch=100,
                batch_size=400,
                shuffle=True,
                validation_data=(encode1_test,encode1_test))


# In[636]:

autoencoder3,encoder3=autoencoder_train(encoded2,'relu','sigmoid',dim[3],dim[2])   #encoder is the predictor
encode2_train=encoder2.predict(encode1_train)
encode2_test=encoder2.predict(encode1_test)
print(encode2_train[0:10])


# In[638]:

early_stopping=EarlyStopping(monitor='val_acc', patience=2)
autoencoder3.fit(encode2_train, encode2_train,
                nb_epoch=100,
                batch_size=400,
                #shuffle=True,
                validation_data=(encode2_test,encode2_test),callbacks=[early_stopping])


# In[640]:

if __name__=='__main__':   
####################用于predict on test#######################
####create a placeholder for an encoded input

    encoded_input = Input(shape=(dim[3],))
    decoded_input1=Input(shape=(dim[2],))
    decoded_input2=Input(shape=(dim[1],))
# retrieve the last layer of the autoencoder model
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
# create the decoder model
    decoder1 = Model(input=encoded_input, output=decoder_layer1(encoded_input))
    decoder2 = Model(input=decoded_input1, output=decoder_layer2(decoded_input1))
    decoder3 = Model(input=decoded_input2, output=decoder_layer3(decoded_input2))

# encode and decode test data
# note that we take them from the *test* set
encoded_imgs = encoder3.predict(encode2_test)
print(encoded_imgs[0:10])
decoded_imgs3 = autoencoder3.predict(encode2_test)
print(decoded_imgs.shape)
    #decoded_imgs2 = decoder2.predict(decoded_imgs1)
    #decoded_imgs3 = decoder3.predict(decoded_imgs2)


# In[641]:

restruct_picture(encode2_test,decoded_imgs3,4)


# In[687]:

if __name__=='__main__':  
######################将股票按收益表现分为五档，排名前0%-20%:0, 20%-40%:1,40%-60%:2,60%-80%:3,80%-100%:4
######################和autoencoder和分类分开训练
    np.random.seed(10)
#假设只有2008年1月份,前三周是训练数据,后一周是测试数据集
    S_day='2008-02-01'#第一个换仓日,本月之前的数据因为因子不全而省略
    L_day='2008-06-04' #最后一个换仓
    Period=20      #回测以周为换仓周期,目前为一周
    classes=2

    Sday=int(split_date(S_day))-1
    C_data=FacData[FacData.Date>Sday]
    
    (TrainX,TrainY),(TestX,TestY)=load_retu('Retu2.csv',C_data,L_day)
    print(TestX.shape)
    print(TrainX.shape)


# In[676]:

#encoded_factors=encoder3.predict(encoder2.predict(encoder1.predict(TrainX))) 
encoded_factors=encoder2.predict(encoder1.predict(TrainX))
encoded_test=encoder2.predict(encoder1.predict(TestX))
print(encoded_factors[0:10])
print(encoded_test[0:10])
#encoded_factors,decoded_factors = predict_autoencoder(automodel,TrainX,LayerDim)   #用于之后的supervised learning 分类
#encoded_test,decoded_test = predict_autoencoder(automodel,TestX,LayerDim)
#print(encoded_factors[0:10])

# convert class vectors to binary class matrices  
Train_Y= np_utils.to_categorical(TrainY, classes)
Test_Y=np_utils.to_categorical(TestY, classes)
print(Test_Y[0:10])
 #在autoencoder的最后一层接入softmax分类器，利用前三周的数据对分类器做训练
softmax=classifier(lr=0.05, decay=1e-6, momentum=0.9,dim=8,cl=classes)




# In[690]:

softmax,loss,accu=train_classifier(softmax, encoded_factors,Train_Y,encoded_test,Test_Y,100)  

#最后一周为测试预测数据 make predictions
#print(Test_Y[0:20])
classes = softmax.predict_classes(encoded_test)
proba = softmax.predict_proba(encoded_test)


#predictions = [float(round(x)) for x in list(probabilities)]
accuracy = np.mean(classes == TestY)
print("\n Prediction Accuracy: %.2f%%" % (accuracy*100))


# In[691]:

print(proba[0:50])
print(classes[0:50])
print(TestY[0:50])
#print([i for i in range(len(classes)) if classes[i]==1])


# In[312]:

print


# In[313]:

print(encode)


# In[ ]:



