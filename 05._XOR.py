'''
AND 0001 OR 0111 NAND 1110 XOR 1001
'''
import numpy as np
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
class LogicGate:
    def __init__(self,gate_name,xdata,tdata):
        self.name=gate_name
        self.__xdata=xdata.reshape(4,2)
        self.__tdata=tdata.reshape(4,1)
        self.__W=np.random.rand(2,1)
        self.__b=np.random.rand(1)
        self.__learning_rate=1e-2
    def __loss_function(self):#__underscores means private in object def
        delta=1e-7
        z=np.dot(self.__xdata,self.__W)+self.__b
        y=sigmoid(z)
        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))
    def error_val(self):#=lossfunction
        delta=1e-7
        z=np.dot(self.__xdata,self.__W)+self.__b
        y=sigmoid(z)
        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))
    def train(self):#f1
        f=lambda x: self.__loss_function()
        print(self.error_val())
        for step in range(8000):
            self.__W-=self.__learning_rate*numerical_derivative(f,self.__W)
            self.__b-=self.__learning_rate*numerical_derivative(f,self.__b)
            if step%400==0:
                print(step,self.error_val(),self.__W,self.__b)
    def predict(self,input_data):
        z=np.dot(input_data,self.__W)+self.__b
        y=sigmoid(z)
        if y>0.5:
            result=1
        else:
            result=0
        return y,result

xdata=np.array([[0,0],[0,1],[1,0],[1,1]])

tdata_AND=np.array([0,0,0,1])
tdata_OR=np.array([0,1,1,1])
tdata_NAND=np.array([1,1,1,0])
tdata_XOR=np.array([0,1,1,0])#안됨 당연히 안됨

AND_obj=LogicGate('ANDGATE',xdata,tdata_AND)
AND_obj.train()
OR_obj=LogicGate('ORGATE',xdata,tdata_OR)
OR_obj.train()
NAND_obj=LogicGate('NANDGATE',xdata,tdata_NAND)
NAND_obj.train()
XOR_obj=LogicGate('XORGATE',xdata,tdata_XOR)
XOR_obj.train()

##for input_data in xdata:
##    sigmoidvalue,logicalvalue=AND_obj.predict(input_data)
##    print(input_data,logicalvalue)

s1=[]
s2=[]
new_input_data=[]
final_output=[]
print(len(xdata))
for index in range(len(xdata)):
    s1=NAND_obj.predict(xdata[index])
    s2=OR_obj.predict(xdata[index])
    new_input_data.append(s1[-1])#s1[-1]=result
    new_input_data.append(s2[-1])
    print(new_input_data)
    sigmoid_val,logical_val=AND_obj.predict(np.array(new_input_data))
    final_output.append(logical_val)
    new_input_data=[]
print(final_output)
for index in range(len(xdata)):
    print(xdata[index],final_output[index])

'''
Sigmoid
ReLU max(0,x)
Leaky RelLU
tanh
'''
#수치미분은 개념적으로만 이해하고 실제로는 오차역전파코드 이
