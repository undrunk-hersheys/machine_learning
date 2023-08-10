'''
1 list/dictionary(keys,values)
2 if 00 in list:
3 xdata=[x[0] for x in rawdata] #열만 출력
  ydata=rawdata[0] #행만 출력
4 def defaultparameter(name,count=2): #count changes when input
    for i in range(count):
  defaultparameter('name',3)
5 lambda
6 class
7 debugging (try,except,finally)
8 with open('test_text1.txt','w') as f:
    f.write('hello python')
with open('test_text1.txt','r') as f:
    print(f.read())

'''

listdata = [10, 20, 30, 40, 50]
dictdata = {'key': 1, 'key2': 2}
if 45 in listdata:
    print('yes')
else:
    print('no')
for listdata in range(0, 5, 2):
    print('0', end='')
print()
list1 = [x**2 for x in range(5)]
print(list1)
rawdata = [[1,2],[3,4],[5.6]]
xdata=[x[0] for x in rawdata]
print(xdata)
evennumber=[]
def sum1(x,y):
    s=x+y
    return s

print(sum1(1,2))


def defaultparameter(name,count=2):
    for i in range(count):
        print(name)
defaultparameter('name')
defaultparameter('name',3)
#adress sharing or data

#lambda for calculus
fx=lambda x: x+100
print(fx(233))


class Person:
    count=0
    def __init__(self,name):
        self.name=name
        print(self.name)
        Person.count+=1
    def work(self,company):
        print('working')
    def getcount(cls):
        return cls.count
#__is only works inside the class
obj1=Person('Name1')
obj2=Person('Name2')
print(Person.count)

def calc(list_data):
    sum1=0
    try:
        sum1=list_data[0]+list_data[1]+list_data[2]
        if sum1<0:
            raise Exception('sum is minus')
    except IndexError as err:
        print(str(err))
    except Exception as err:
        print(str(err))
    finally:
        print(sum1)
calc([1,2])
calc([1,2,-100])

with open('test_text1.txt','w') as f:
    f.write('hello python')
with open('test_text1.txt','r') as f:
    print(f.read())

#pip install numpy
#pip install pandas
#pip install matplotlib
#pip install tensorflow
