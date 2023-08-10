import numpy as np
loaded_data=np.loadtxt('logistic.csv',delimiter=',',dtype=np.float32)
xdata=loaded_data[:,0:-1]
tdata=loaded_data[:,[-1]]
rows,columns=xdata.shape
W=np.random.rand(columns,1)
b=np.random.rand(1,1)
#print(xdata,'\n',tdata)
print(loaded_data)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss_function2(X,t):#error value 와 차이를 두는 이유는 명시적으로
    #사용자가 알 수 있도록 하기 위해서이다.
    delta_log=1e-8
    z=np.dot(X,W)+b
    #y=(1/(1+np.exp(-z)))
    y=sigmoid(z)
    return -np.sum(t*np.log(y+delta_log)+(1-t)*np.log(1-y+delta_log))
##f1=lambda x: loss_function2(xdata,tdata)
def f1(x):
    a=loss_function2(xdata,tdata)
    return a
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
learning_rate=1e-4
for step in range(20000):
    W-=learning_rate*numerical_derivative(f1,W)
    b-=learning_rate*numerical_derivative(f1,b)
    if step%800==0:
        print(step,W,b,f1(1))
def predict(x):
    z=np.dot(x,W)+b
    y=(1/(1+np.exp(-z)))
    if y>0.5:
        result=1
    else:
        result=0
    return y,result
print(predict([-3,1]))
print(predict([-10,-10]))
