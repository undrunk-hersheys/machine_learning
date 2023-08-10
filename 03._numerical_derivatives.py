'''
1 numerical derivatives
'''
def numerical_derivative(f,x):
    delta_x=1e-4
    return (f(x+delta_x)-f(x-delta_x))/(2*delta_x)
ff=lambda x: x**2
a=3
print(numerical_derivative(ff,a))
print(1e-4)

##다변수함수 편미분
import numpy as np
def numerical_derivative(f,x): #x contains lots of variables
    delta_x=1e-4
    grad=np.zeros_like(x) #similar to zeros
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index

        print(it)
        print(idx)

        tmp_val=x[idx]
        x[idx]=float(tmp_val)+delta_x
        fx1=f(x)
        x[idx]=float(tmp_val)-delta_x
        fx2=f(x)
        grad[idx]=(fx1-fx2)/(2*delta_x)
        x[idx]=tmp_val
        it.iternext() #to the next it
    return grad

func1=lambda z:z[0]*2+3*z[0]*z[1]
list1=np.array([1.0,2.0])

func2=lambda z:z[0]**2
list2=np.array([3.0])

def func3(input_obj):
    w=input_obj[0,0]
    x=input_obj[0,1]
    y=input_obj[1,0]
    z=input_obj[1,1]
    return (w*x+x*y*z+3*w+z*np.power(y,2))
list3=np.array([[1.0,2.0],[3.0,4.0]]) #no different with vector

n_d=numerical_derivative(func3,list3)
print(np.array(n_d))


