'Hello MNIST'
#from keras.datasets import mnist
import numpy as np
training_data=np.loadtxt('./mnist_train.csv',delimiter=',')
test_data=np.loadtxt('./mnist_test.csv',delimiter=',')
print(training_data.shape,test_data.shape)#1 row number 2~n row color data

##import matplotlib.pyplot as plt
##img=training_data[10][1:].reshape(28,28)
##plt.imshow(img,cmap='gray')
##plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))
def numerical_derivative(f,w):
    delta=1e-4
    grad=np.zeros_like(w)
    it=np.nditer(w,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        tmp=w[idx]
        w[idx]=float(tmp)+delta
        fx1=f(w)
        w[idx]=float(tmp)-delta
        fx2=f(w)
        grad[idx]=(fx1-fx2)/(2*delta)
        w[idx]=tmp
        it.iternext()
    return grad

class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        #2층
        self.W2=np.random.rand(self.input_nodes,self.hidden_nodes)#nod 6개로 임의로 설정
        self.b2=np.random.rand(self.hidden_nodes)
        #3층
        self.W3=np.random.rand(self.hidden_nodes,self.output_nodes)
        self.b3=np.random.rand(self.output_nodes)
        #learning rate reset
        self.learning_rate=1e-3
        
    def feed_forward(self):
        delta=1e-7
        z1=np.dot(self.input_data,self.W2)+self.b2#은닉층1
        a1=sigmoid(z1)
        z2=np.dot(a1,self.W3)+self.b3#출력층
        y=a2=sigmoid(z2)
        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))
###feed foward 가 lossfunction 대신
##    def __loss_function(self):#__underscores means private in object def
##        delta=1e-7
##        z=np.dot(self.__xdata,self.__W)+self.__b
##        y=sigmoid(z)
##        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))
    def loss_val(self):
        delta=1e-7
        z1=np.dot(self.input_data,self.W2)+self.b2#은닉층1
        a1=sigmoid(z1)
        z2=np.dot(a1,self.W3)+self.b3#출력층
        y=a2=sigmoid(z2)
        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))

    def train(self,training_data):#f1
        self.target_data=np.zeros(output_nodes)+0.01
        self.target_data[int(training_data[0])]=0.99#T 즉 t는 10의 벡터로 나타내는 것
        self.input_data=(training_data[1:]/255.0*0.99)+0.01#prevent from overflow #X
        f=lambda x: self.feed_forward()
        self.W2-=self.learning_rate*numerical_derivative(f,self.W2)
        self.b2-=self.learning_rate*numerical_derivative(f,self.b2)
        self.W3-=self.learning_rate*numerical_derivative(f,self.W3)
        self.b3-=self.learning_rate*numerical_derivative(f,self.b3)
    def predict(self,input_data):
        z1=np.dot(input_data,self.W2)+self.b2
        a1=sigmoid(z1)
        z2=np.dot(a1,self.W3)+self.b3
        y=a2=sigmoid(z2)
        predict=np.argmax(y)
        return predict
    def accuracy(self,test_data):
        matched_list=[]
        not_matched_list=[]
        for index in range(len(test_data)):
            label=int(test_data[index,0])
            data=(test_data[index,1:]/255*0.99)+0.01
            predicted_num=self.predict(data)
            if label==predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print(100*(len(matched_list)/len(test_data)))
        return matched_list,not_matched_list

input_nodes=784
hidden_nodes=5
output_nodes=10

nn=NeuralNetwork(input_nodes,hidden_nodes,output_nodes)
for step in range(3000):
    #index=np.random.randint(0,len(training_data)-1)#supposed to be 60000 not 3000
    index=step+2
    nn.train(training_data[index])
    if step%300==0:
        print(step,nn.loss_val())
nn.accuracy(test_data)
