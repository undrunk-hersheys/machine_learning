'''
ORGANIZING NEW
1 import numpy as np
2 np.array is showing row column stuff(vector,matrix)
3 a.shape is showing row*column (0,0)
4 a.reshape(0,0) changes #it pushes atoms to make structure
#=np.reshape(a,())
#=np.arrange().reshape([])
5 ndim shows dimension [[[]]]=3
6 c=np.dot(a,b) 행렬곱
7 b=a.T reverse row column 전치행렬
8 b=a[:,:] slicing row column
9 shows all atoms
it=np.nditer(j,flags=['multi_index'],op_flags=['readwrite'])
while not it.finished:
    idx=it.multi_index
    print(j[idx])
    it.iternext()
10 c=np.concatenate((a,b),axis=0)
0 is adding row 1 is adding column
have to reshape to 1,n or n,1 
11 loaded_data=np.loadtxt('./data-01.csv', delimiter=',',dtype=np.float32)
12 a=np.random.rand(3,4) #4*3행렬
b=np.random.randint(5,10,size=(3,4)) #for int
13 sum,exp,log,max,min,argmax,argmin
print(np.sum(z),'\n',np.exp(z),'\n',np.log(z))
print(np.max(z),np.min(z),np.argmax(z),np.argmin(z)) +mean,median
print(np.argmax(z,axis=0))
print(np.argmax(z,axis=1))
#axis=0 다른 행과 비교,열기준  axis=1 다른 열과 비교,행기준
14 a=np.ones/zeros([3,3]) all 0 or all 1
15 import matplotlib
'''




import numpy as np
a=np.array([1,2]) #vector
print(a,type(a))
#most common way of doing it

##from numpy import exp
##result=exp(1)
##print('result==',result,',type==',type(result))

#numpy vector matrix
#dot product

a=[[1,0],[0,1]]
b=[[1,1],[1,1]]
print(a+b)

a=np.array([[1,0],[0,1]])
b=np.array([[1,1],[1,1]])
print(a+b)

print(a.shape,b.shape)
#2*2 or 3*3 etc row*column vertical*horizontal

print(a.ndim,b.ndim)
#shows whether 1d 2d 3d

a.reshape(2,2)
#change column and row

#행렬곱은 a의 열과 b의 행이 같아야만 가능
#그래서 reshape을 하거나 전치행렬등을 사용해야만함

c=np.array([[1,2,3],[4,5,6]])
d=np.array([[-1,-2],[-3,-4],[-5,-6]])
e=np.dot(c,d)#행렬곱
print(c.shape,d.shape)
print(c.ndim,d.ndim)
print(d.ndim)
print(e)

#broadcast 사칙연산
#상수일 경우 그 덧셈에 맞춰 같은 크기의 행열로 생성
#in python automatically apply this broadcast

#전치행렬 행-열 열-행 변경 transpose
f=c.T #T in capital
print(f)
print(c)

g=np.array([1,2,3,4,5,6])
print(g)
print(g.ndim)
h=np.array([1,2,3,4,5,6,7,8,9])
print(h.ndim)
print('llllllllllllllllll')
i=g.reshape(3,2)
print(i)
#slicing also share the adress so when change in sliced one changes origin
print(i[0:-1,1:2])
print(i[:,0])
print(i[:,:])
#i[row,column]

#iterator
#순차적으로 읽어드리는 기능
j=np.array([[1,2,3,4],[5,6,7,8]])
print(j,'\n',j.shape)
#행렬원소접근
it=np.nditer(j,flags=['multi_index'],op_flags=['readwrite'])
while not it.finished:
    idx=it.multi_index
    print(j[idx], end=' ')
    #print(j)
    it.iternext()
print()
'''
plus googling
a=np.arrange(0,60,5)
a=a.reshape(4,3)
for x in np.nditer(a,order='C'): #행 순서대로 출력
    print x
for x in np.nditer(a,order='F'): #열 순서대로 출력
    print x

for x in np.nditer(a,op_flags=[readwrite']):
    x[...]=2*x
    print(a)

#flags=[c_index,f_index,multi_index,external_loop]
c_index=행 출력
f_index=열 출력
multi_index=순서 출력 1,2,3,4,5...
external_loop=[index] 출력으로 표현
'''


'''
k=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,0],[11,22]]])
print(k)
#way of showing 3d
'''

#adding new row column
#numpy concatenate
l=np.array ([[1,2,3],[4,5,6,]])
row_add=np.array([7,8,9]).reshape(1,3,)
column_add=np.array([100,200]).reshape(2,1)
print(column_add.shape)
m=np.concatenate((l,column_add),axis=1)
print(m)

##import random
##file1=open('data-01.txt','w')
##for k in range(25):
####    a=str(random.randint(1,100))
####    b=str(random.randint(1,100))
####    c=str(random.randint(1,100))
####    d=str(random.randint(1,100))
####    file1.write(a,end=',')
####    file1.write(b,end=',')
####    file1.write(c,end=',')
####    file1.write(d)
####    print()
##    file1.write([1,2,3,4])
##file1.close()

    
        
#loading data
loaded_data=np.loadtxt('./data-01.csv', delimiter=',',dtype=np.float32)
#float 32 is for the 32bit also can work when 62bit
#dtype=np.float32 생략가능
#행의 구분 enter, 열의 구분 delitmiter 에서 표시한 문자로 구
x=loaded_data[:,0:-1]
y=loaded_data[:,[-1]]
print(x)
print(y)

#임의설정시 유용
#행렬을 랜덤으로 설정
random_number1=np.random.rand(3)
print(random_number1,random_number1.shape)
random_number2=np.random.rand(3,1)
print(random_number2,random_number2.shape)

z=np.array([[2,3,4,5],[44,55,66,77],[5,6,7,8],[9,11,22,33]])
print(np.sum(z),'\n',np.exp(z),'\n',np.log(z))
print(np.max(z),np.min(z),np.argmax(z),np.argmin(z))
#argmax=index(주소)를 줌 max의 주소 3 min의 주소 0
#works in 2dimension
print(np.argmax(z,axis=0))
print(np.argmax(z,axis=1))
#axis=0 열기준 axis=1 행기준

one=np.ones([3,3])
print(one.shape,one)
zero=np.zeros([3,2])
print(one.shape,zero)
#해당행열을 모두 0 1로 채워줌

#matplotlib- drawing graph
import matplotlib.pyplot as plt
##x_data=np.random.rand(100)
##y_data=np.random.rand(100)
##plt.title('ScatterPlot') #title
##plt.grid() #presence or absence of gird
##plt.scatter(x_data,y_data,color='b',marker='o') #color and marker
##plt.show()
x_data=[x**2 for x in range(-10,10)]
y_data=[y for y in range(-10,10)]
plt.grid()
plt.plot(x_data,y_data,color='b')
plt.show()
#in jupiter %matplotlib inline 추가로 적어줘야

'''
추가 googling 정보들
np.empty( int or tuple )
메모리의 상태에 따라 랜덤으로 배열을 만듭니다.
나중에 요소를 수정하거나 더 채워 넣을 수 있습니다.
처리속도를 높이기 위해 사용하는 기법입니다.
np.arange( scalar )
파이썬에서 사용하는 range() 함수의 결과가 요소로 들어갑니다
range( ) 에서 사용 가능한 기법들
( 첫 번째 숫자 지정, 두 번째 숫자 지정, step size 지정) 역시 사용 가능합니다.
np.linspace( scalar, num=n )
간격을 지정하여 선형으로 배치된 n개의 값을 갖는 배열을 만들 수 있습니다.
eye( n )
n차 단위 행렬(n차 정사각 행렬에서 대각선의 원소가 모두1이고,
다른 원소는 모두 0인 행렬을 의미합니다.
단위행렬 E를 임의의 행렬 A와 곱하면 행렬 A가 얻어집니다.)을 생성하는 함수입니다.
참고로 단위행렬이란 n차 정사각행렬에서 대각선의 원소가 모두 1이고,
다른 원소는 모두 0인 행렬을 의미합니다.
단위행렬 E를 임의의 행렬 A와 곱하면 행렬 A가 얻어집니다.
'''
