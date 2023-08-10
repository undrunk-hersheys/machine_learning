import numpy as np
load_data=np.loadtxt('data-01.csv',delimiter=',')#,dtype=np.float32)
z=load_data[:,:]
raw_data=np.array([[1,1,1,3],[2,2,2,6],[3,3,3,9],[4,4,4,12],[5,5,5,15]])
rows,columns=raw_data.shape
print(rows,columns)
x1data=raw_data[:,[0]]
x2data=raw_data[:,[1]]
x3data=raw_data[:,[2]]
xdata=raw_data[:,0:-1]
print(len(xdata))
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
for step in range(8000):
    W-=learning_rate*numerical_derivative(f,W)
    b-=learning_rate*numerical_derivative(f,b)
    if step%800==0:
        print(step,loss_function(xdata,tdata),W,b)
def predict(x):
    y=np.dot(x,W)+b
    return y
print(predict([6,6,6]))
