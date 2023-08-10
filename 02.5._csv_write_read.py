import csv
f=open('testing.csv','w',newline='')
wr=csv.writer(f)
wr.writerow([1,2,3,'q'])
wr.writerow([4,5,6,'w'])
f.close()

f=open('testing.csv','r')
rdr=csv.reader(f)
for line in rdr:
    print(line)
f.close()

import numpy as np
a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
np.savetxt('testing2.csv',a,delimiter=',')

loaded_data=np.loadtxt('testing2.csv', delimiter=',',dtype=np.float32)
#dtype=np.float32 no need to write
x=loaded_data[:,:]
y=loaded_data[:,:]
print(x)
print(y)
