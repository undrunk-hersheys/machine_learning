'''
지도학습 supervised learning 회귀 regression 분류 classification <<
비지도학습 unsupervised learning 군집화 clustering

regression 연속된 숫자값 예측
classification 특정값 분류

clustering 특정값끼리 묶어서 그룹화
'''
'''
Linear Regression
y=Wx+b (weight, bias)
error=t-y=t-(Wx+b) #t=data y/x is given
loss function(cost function)
(t-y)**2=(t-[Wx+b])**2
sigma (t-[Wx+b])**2/n #the getting variable is W and b
'''
'''
경사하강법 gradient decent algorithm
update W and b as W=W-a*(dE(W,b)/dW) repeat until dE(W,b)/dW=0 also b
a is learning rate = 1e-3 or 1e-4 or 1e-5
행렬곱 사용 X(n*1) W(1*1) => n*1
'''
'''
single_variable regression
'''
import numpy as np
##x_data=np.array([1,2,3,4,5]).reshape(5,1)
##t_data=np.array([2,3,4,5,6]).reshape(5,1)

raw_data=np.array([[-3,9],[-2,4],[-1,1],[1,1],[2,4],[3,9],[4,16],[5,25]])
xdata=raw_data[:,[0]]
tdata=raw_data[:,[1]]
print(xdata,'\n',tdata)

W=np.random.rand(1,1)
b=np.random.rand(1)
print(W,W.shape,b,b.shape)

def loss_function(x,t):
    y=np.dot(x,W)+b
    return (np.sum((t-y)**2))/len(x)
def predict(x):
    y=np.dot(x,W)+b
    return y

#from 03_numerical_derivatives.py
def numerical_derivative(f,x): #x contains lots of variables
    delta_x=1e-4
    grad=np.zeros_like(x) #similar to zeros
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        
##        print(idx)

        tmp_val=x[idx]
        x[idx]=float(tmp_val)+delta_x
        fx1=f(x)
        x[idx]=float(tmp_val)-delta_x
        fx2=f(x)
        grad[idx]=(fx1-fx2)/(2*delta_x)
        x[idx]=tmp_val
        it.iternext() #to the next it
    return grad

learning_rate=1e-2
f=lambda x: loss_function(xdata,tdata)
print(loss_function(xdata,tdata))

for step in range(10000):
    W-=learning_rate*numerical_derivative(f,W)
    b-=learning_rate*numerical_derivative(f,b)
    if step%400==0:
        print(step,loss_function(xdata,tdata),W,b)
for tstn in range(-10,10):
    print(tstn, predict([int(tstn)]))
'''
multi_variable regression
x1,x2,x3,t,W1,W2,W3
'''
load_data=np.loadtxt('data-01.csv',delimiter=',',dtype=np.float32)
z=load_data[:,:]
raw_data=np.array([[1,1,1,3],[2,2,2,6],[3,3,3,9],[4,4,4,12],[5,5,5,15]])
x1data=raw_data[:,[0]]
x2data=raw_data[:,[1]]
x3data=raw_data[:,[2]]
xdata=raw_data[:,0:-1]
tdata=raw_data[:,[3]]
print(xdata,'\n',tdata)
W=np.random.rand(3,1)#0<=,<1
b=np.random.rand(1)
print(W,W.shape,b,b.shape)
def loss_function(x,t):
    y=np.dot(x,W)+b
    return (np.sum((t-y)**2))/len(x)
def numerical_derivative(f,x):
    delta_x=1e-4
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        tmp_val=x[idx]
        x[idx]=float(tmp_val)+delta_x
        fx1=f(x)
        x[idx]=float(tmp_val)-delta_x
        fx2=f(x)
        grad[idx]=(fx1-fx2)/(2*delta_x)
        x[idx]=tmp_val
        it.iternext()
    return grad
learning_rate=1e-2
f=lambda x: loss_function(xdata,tdata)
print(loss_function(xdata,tdata))
for step in range(1):
    W-=learning_rate*numerical_derivative(f,W)
    b-=learning_rate*numerical_derivative(f,b)
    if step%400==0:
        print(step,loss_function(xdata,tdata),W,b)
def predict(x):
    y=np.dot(x,W)+b
    return y
print(predict([6,6,6]))
'''
logistic regression
'''

