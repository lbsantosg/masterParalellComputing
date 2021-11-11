import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
line = input()
print(line)
idx = line.split(' ')[0]
name = line.split(' ')[1].split('.')[0] + "_plot.png"
data = pd.read_csv("Time_file_"+idx+".csv", delimiter = ",")
#print(data)
averages = []
speedup = []
secuential = 1 
for t in data.index:
  av = 0 
  for tries in data[['Try_1','Try_2','Try_3']]:
    av += data.loc[t,tries]
  averages.append((av/3.0))
  if (data.loc[t]['Threads'] == 1.0):
    secuential = av/3.0
  speedup.append(secuential/(av/3.0))
data["Average"] = averages
data["Speedup"] = speedup
print(data)
kernel_sz3 = data[data['Kernel_size'] == 3]
kernel_sz5 = data[data['Kernel_size'] == 5]
kernel_sz7 = data[data['Kernel_size'] == 7]
kernel_sz9 = data[data['Kernel_size'] == 9]
kernel_sz11 = data[data['Kernel_size'] == 11]
kernel_sz13 = data[data['Kernel_size'] == 13]
kernel_sz15 = data[data['Kernel_size'] == 15]
plt.plot(kernel_sz3['Threads'],kernel_sz3['Average'], color='red', label = 'Kernel 3')
plt.plot(kernel_sz5['Threads'],kernel_sz5['Average'], color='blue', label = 'Kernel 5')
plt.plot(kernel_sz7['Threads'],kernel_sz7['Average'], color='green', label = 'Kernel 7')
plt.plot(kernel_sz9['Threads'],kernel_sz9['Average'], color='pink', label = 'Kernel 9')
plt.plot(kernel_sz11['Threads'],kernel_sz11['Average'], color='purple', label = 'Kernel 11')
plt.plot(kernel_sz13['Threads'],kernel_sz13['Average'], color='yellow', label = 'Kernel 13')
plt.plot(kernel_sz15['Threads'],kernel_sz15['Average'], color='orange', label = 'Kernel 15')
plt.xlabel("# of threads")
plt.ylabel("Seconds")
plt.title("Time vs Threads (" + line.split(' ')[1]+ ") using OMP")
plt.legend()
plt.savefig('schedule_time' + name)
#plt.show()
plt.close()
plt.plot(kernel_sz3['Threads'],kernel_sz3['Speedup'], color='red', label = 'Kernel 3')
plt.plot(kernel_sz5['Threads'],kernel_sz5['Speedup'], color='blue', label = 'Kernel 5')
plt.plot(kernel_sz7['Threads'],kernel_sz7['Speedup'], color='green', label = 'Kernel 7')
plt.plot(kernel_sz9['Threads'],kernel_sz9['Speedup'], color='pink', label = 'Kernel 9')
plt.plot(kernel_sz11['Threads'],kernel_sz11['Speedup'], color='purple', label = 'Kernel 11')
plt.plot(kernel_sz13['Threads'],kernel_sz13['Speedup'], color='yellow', label = 'Kernel 13')
plt.plot(kernel_sz15['Threads'],kernel_sz15['Speedup'], color='orange', label = 'Kernel 15')
plt.xlabel("# of threads")
plt.ylabel("Speedup")
plt.title("Time vs Speedup (" +line.split(' ')[1]+ ") using OMP")
plt.legend()
plt.savefig('schedule_speedup' + name)
#plt.show()