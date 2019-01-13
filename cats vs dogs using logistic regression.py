import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy import misc
import random

def main():
        trainfiles=os.listdir("train")
        tr=[]
        train=[]
        for i,im in enumerate(trainfiles):
            filename='train/'+im
            image=np.asarray(imread(filename))
            tr.append((image,1 if im.split(".")[0]=='cat' else 0))
        for i in range(len(tr)):
            a=random.randint(0,len(tr)-1)
            train.append(tr[a])
            
        
        
        trainsize=int(len(trainfiles)*0.8)
        train_x=np.zeros((trainsize,64,64,3))
        train_y=np.zeros((1,trainsize))
        valid_x =np.zeros((len(trainfiles)-trainsize,64,64,3))
        valid_y=np.zeros((1,len(trainfiles)-trainsize))
        
        import warnings
        
        warnings.simplefilter("ignore",DeprecationWarning)
        for i,(x,y) in enumerate(train):
            reszd=misc.imresize(x,(64,64,3))
            if(i<trainsize):
                train_x[i]=reszd
                train_y[:,i]=y
            else:
                valid_x[i-trainsize]=reszd
                valid_y[:,i-trainsize]=y
        print(np.shape(train_x))
        print(train_x.shape[0])
        print(valid_y.shape)
        print("                                                       ")
        flatten_x_train=train_x.reshape(train_x.shape[0],-1)
        flatten_valid_x=valid_x.reshape(valid_x.shape[0],-1)
        flatten_x_train=flatten_x_train/255
        flatten_valid_x=flatten_valid_x/255
        
        w=np.zeros((64*64*3,1))
        b=0
        cost=0
        train_y=train_y.reshape(train_y.shape[1],1)
        valid_y=valid_y.reshape(valid_y.shape[1],1)
        print(w.shape)
        print(flatten_x_train.shape)
        cost=0
        b=0
        xaxis=[]
        for i in range(100000):
            w,b=calgrad(flatten_x_train,train_y,w,b)
            cost=calcost(cost,w,flatten_x_train,b,train_y)
            if(i%100==0):
                xaxis.append(cost)
            
        print(cost)
        plotsig(flatten_x_train,w,b)
        plot(xaxis)
        predict(flatten_x_train,train_y,w,b)
        predict(flatten_valid_x,valid_y,w,b)
            
 
def plot(x) :
    b=range(100000)
    plt.plot(b,x)
    plt.show()
    

def plotsig(x,w,b):
    sig=1/(1+(np.exp(-(np.dot(x,w)+b))))
    z=np.dot(x,w)+b
    plt.plot(z,sig)
    plt.show()
            
def predict(x,y,w,b):
       sig=1/(1+(np.exp(-(np.dot(x,w)+b))))
       print("test accuracy: {} %".format(100 - np.mean(np.abs(sig - y)) * 100))
       testimage('C:/Users/venu/Desktop/test1/1553.jpg',w,b)
       
              
               
               
              
            
            
def calcost(cost,w,x,b,y):
    m=x.shape[0]
    A = 1/(1+np.exp(-(np.dot(x, w) + b)))
    cost = (- 1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A)))
    return cost

def calgrad(x,y,w,b):
    alph=0.005
    m=x.shape[0]
    sig= 1/(1+np.exp(-(np.dot(x, w) + b)))
    w =w-alph*((1 / m) * np.dot(x.T, (sig - y)))
    b =b-alph*((1 / m) * np.sum(sig - y))

    return w,b
    

def testimage(img,w,b):
    
    image=np.asarray(imread(img))
    reszd=misc.imresize(image,(64,64,3))
    x=reszd.reshape(1,-1)
    x=x/255
    sig=1/(1+(np.exp(-(np.dot(x,w)+b))))
    print(sig)
    if(sig>=0.5):
        print('cat')
    else:
        print('dog')
    
    
    
main()





    
    




