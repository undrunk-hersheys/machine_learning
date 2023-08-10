#backpropagation 오차역전파 line_regression
import numpy as np
training_data=np.loadtxt('./mnist_train.csv',delimiter=',')
test_data=np.loadtxt('./mnist_test.csv',delimiter=',')
print(training_data.shape,test_data.shape)
#print(len(training_data))

def sigmoid(z):
    return 1/(1+np.exp(-z))

from datetime import datetime

class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        #은닉층(784*100)
        self.W2=np.random.rand(self.input_nodes,self.hidden_nodes)/np.sqrt(self.input_nodes/2)
        self.b2=np.random.rand(self.hidden_nodes)
        #출력층(100*10)
        self.W3=np.random.rand(self.hidden_nodes,self.output_nodes)/np.sqrt(self.hidden_nodes/2)
        self.b3=np.random.rand(self.output_nodes)
        #선형회귀
        self.Z3=np.zeros([1,output_nodes])
        self.A3=np.zeros([1,output_nodes])
        self.Z2=np.zeros([1,hidden_nodes])
        self.A2=np.zeros([1,hidden_nodes])
        self.Z1=np.zeros([1,input_nodes])
        self.A1=np.zeros([1,input_nodes])
        #학습률
        self.learning_rate=learning_rate
    def feed_forward(self):
        delta=1e-7
        self.Z1=self.input_data
        self.A1=self.input_data
        self.Z2=np.dot(self.A1,self.W2)+self.b2
        self.A2=sigmoid(self.Z2)
        self.A3=np.dot(self.A2,self.W3)+self.b3
        #self.A3=sigmoid(self.Z3)
        return np.sum((self.target_data-self.A3)*(self.target_data-self.A3))/784
    def loss_val(self):
        delta=1e-7
        self.Z1=self.input_data
        self.A1=self.input_data
        self.Z2=np.dot(self.A1,self.W2)+self.b2
        self.A2=sigmoid(self.Z2)
        self.A3=np.dot(self.A2,self.W3)+self.b3
        #self.A3=sigmoid(self.Z3)
        return np.sum((self.target_data-self.A3)**2)/784
    def train(self,input_data,target_data):#f1
        self.target_data=target_data
        self.input_data=input_data
        loss_val=self.feed_forward()
        loss_3=(self.A3-self.target_data)#loss_2,_3 all vector function can't -=
        self.W3-=self.learning_rate*np.dot(self.A2.T,loss_3)
        self.b3=self.b3-self.learning_rate*loss_3
        loss_2=np.dot(loss_3,self.W3.T)#erased sigmoid derivative
        self.W2=self.W2-self.learning_rate*np.dot(self.A1.T,loss_2)
        self.b2=self.b2-self.learning_rate*loss_2
    def predict(self,input_data):
        A2=np.dot(input_data,self.W2)+self.b2
        #A2=sigmoid(Z2)
        A3=np.dot(A2,self.W3)+self.b3
        #A3=sigmoid(Z3)
        #predict=np.argmax(A3)
        return A3
    def accuracy(self,test_data):
        matched_list=[]
        not_matched_list=[]
        for index in range(len(test_data)):
            label=int(test_data[index,0])
            data=(test_data[index,1:]/255*0.99)+0.01
            predicted_num=self.predict(np.array(data,ndmin=2))
##            if label==predicted_num:
##                matched_list.append(index)
##            else:
##                not_matched_list.append(index)
            #if int(predicted_num)==label:
            if label-0.5<=predicted_num<label+0.5:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print(100*(len(matched_list)/len(test_data)))
        return matched_list,not_matched_list

input_nodes=784
hidden_nodes=10 #100:91.5 50:88.71 3:48.36 2:35.76
output_nodes=1 #onenode
learning_rate=0.3
epochs=1
nn=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

start_time=datetime.now()

for i in range(epochs):
    for step in range(len(training_data)):
        target_data=np.zeros(output_nodes)+0.01
        target_data[0]=int(training_data[step,0])
        #target_data=np.array(training_data[step,0])
        input_data=((training_data[step,1:]/255.0)*0.99)+0.01
        nn.train(np.array(input_data,ndmin=2),np.array(target_data,ndmin=2))
        #nn.train(input_data,target_data)
        if step%4000==0:
            print(step,nn.loss_val())
nn.accuracy(test_data)

end_time=datetime.now()
print(end_time-start_time)


##Warning (from warnings module):
##  File "C:\Users\user\Documents\코딩\Python\machine learning\07._back_propagation_MNIST_line_regression.py", line 41
##    return np.sum(self.target_data-self.A3)**2
##RuntimeWarning: overflow encountered in double_scalars
##
##Warning (from warnings module):
##  File "C:\Users\user\Documents\코딩\Python\machine learning\07._back_propagation_MNIST_line_regression.py", line 39
##    self.A3=np.dot(self.A2,self.W3)+self.b3
##RuntimeWarning: overflow encountered in add
##
##Warning (from warnings module):
##  File "C:\Users\user\Documents\코딩\Python\machine learning\07._back_propagation_MNIST_line_regression.py", line 56
##    self.W3-=self.learning_rate*np.dot(self.A2.T,loss_3)
##RuntimeWarning: invalid value encountered in subtract
##
##Warning (from warnings module):
##  File "C:\Users\user\Documents\코딩\Python\machine learning\07._back_propagation_MNIST_line_regression.py", line 57
##    self.b3=self.b3-self.learning_rate*loss_3
##RuntimeWarning: invalid value encountered in subtract

