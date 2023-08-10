import numpy as np
load_data=np.loadtxt('testing3.csv',delimiter=',',dtype=np.float32)
z=load_data[:,:]
rows,columns=z.shape
print(rows,columns)
xdata=z[:,0:-1]
tdata=z[:,[-1]]
print(xdata,'\n',tdata)
W1=np.random.rand(1,1)#0<=,<1
W2=np.random.rand(1,1)
b=np.random.rand(1)
print(W1,W2,b)
def loss_function(x,t):
    y=np.dot(x,W1)+np.dot(x*x,W2)+b
    return (np.sum((t-y)**2))/len(x)
f=lambda x: loss_function(xdata,tdata)
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
print(loss_function(xdata,tdata))
learning_rate=1e-5
for step in range(80000):
    W1-=learning_rate*numerical_derivative(f,W1)
    W2-=learning_rate*numerical_derivative(f,W2)
    b-=learning_rate*numerical_derivative(f,b)
    if step%8000==0:
        print(step,loss_function(xdata,tdata),W1,W2,b)
def predict(x):
    y=np.dot(x,W1)+np.dot(x*x,W2)+b
    return y
print(predict(np.array([10])))

