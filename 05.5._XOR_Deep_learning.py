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
        self.__xdata=xdata.reshape(4,2)#batch 한번에 계산가능하게 하는 것
        self.__tdata=tdata.reshape(4,1)
        #2층
        self.__W2=np.random.rand(2,6)#nod 6개로 임의로 설정
        self.__b2=np.random.rand(6)
        #3층
        self.__W3=np.random.rand(6,1)
        self.__b3=np.random.rand(1)
        #learning rate reset
        self.__learning_rate=1e-2
    def feed_forward(self):
        delta=1e-7
        z2=np.dot(self.__xdata,self.__W2)+self.__b2#은닉층1
        a2=sigmoid(z2)
        z3=np.dot(a2,self.__W3)+self.__b3#출력층
        y=a3=sigmoid(z3)
        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))

###feed foward 가 lossfunction 대신
##    def __loss_function(self):#__underscores means private in object def
##        delta=1e-7
##        z=np.dot(self.__xdata,self.__W)+self.__b
##        y=sigmoid(z)
##        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))
##    def error_val(self):#=lossfunction
##        delta=1e-7
##        z=np.dot(self.__xdata,self.__W)+self.__b
##        y=sigmoid(z)
##        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))

    def loss_val(self):
        delta=1e-7
        z2=np.dot(self.__xdata,self.__W2)+self.__b2#은닉층1
        a2=sigmoid(z2)
        z3=np.dot(a2,self.__W3)+self.__b3#출력층
        y=a3=sigmoid(z3)
        return -np.sum(self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log((1-y)+delta))
    def train(self):#f1
        f=lambda x: self.feed_forward()
        print(self.loss_val())
        for step in range(10000):
            self.__W2-=self.__learning_rate*numerical_derivative(f,self.__W2)
            self.__b2-=self.__learning_rate*numerical_derivative(f,self.__b2)
            self.__W3-=self.__learning_rate*numerical_derivative(f,self.__W3)
            self.__b3-=self.__learning_rate*numerical_derivative(f,self.__b3)
            if step%400==0:
                print(step,self.loss_val())
    def predict(self,input_data):
        z2=np.dot(input_data,self.__W2)+self.__b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.__W3)+self.__b3
        y=a3=sigmoid(z3)

        if y>0.5:
            result=1
        else:
            result=0
        return y,result

xdata=np.array([[0,0],[0,1],[1,0],[1,1]])

##tdata_AND=np.array([0,0,0,1])
##tdata_OR=np.array([0,1,1,1])
##tdata_NAND=np.array([1,1,1,0])
tdata_XOR=np.array([0,1,1,0])#안됨 당연히 안됨

##AND_obj=LogicGate('ANDGATE',xdata,tdata_AND)
##AND_obj.train()
##OR_obj=LogicGate('ORGATE',xdata,tdata_OR)
##OR_obj.train()
##NAND_obj=LogicGate('NANDGATE',xdata,tdata_NAND)
##NAND_obj.train()
XOR_obj=LogicGate('XORGATE',xdata,tdata_XOR)
XOR_obj.train()

for input_data in xdata:
    sigmoidvalue,logicalvalue=XOR_obj.predict(input_data)
    print(input_data,logicalvalue)

##s1=[]
##s2=[]
##new_input_data=[]
##final_output=[]
##print(len(xdata))
##for index in range(len(xdata)):
##    s1=NAND_obj.predict(xdata[index])
##    s2=OR_obj.predict(xdata[index])
##    new_input_data.append(s1[-1])#s1[-1]=result
##    new_input_data.append(s2[-1])
##    print(new_input_data)
##    sigmoid_val,logical_val=AND_obj.predict(np.array(new_input_data))
##    final_output.append(logical_val)
##    new_input_data=[]
##print(final_output)
##for index in range(len(xdata)):
##    print(xdata[index],final_output[index])

#수치미분은 개념적으로만 이해하고 실제로는 오차역전파코드 이
