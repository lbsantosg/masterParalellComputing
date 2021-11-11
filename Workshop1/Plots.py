import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
line = input()
print(line)
idx = line.split(' ')[0]
name = line.split(' ')[1].split('.')[0] + "_plot.png"
data = pd.read_csv(idx, delimiter = ",")
#print(data)
averages = []
speedup = []
idx_after = 0
idx=0
for t in data.index:
  av = 0 
  for tries in data[['Try_1','Try_2','Try_3']]:
    av += data.loc[t,tries]
  averages.append((av/3.0))
  speedup.append(averages[idx-idx_after]/(av/3.0))
  idx = idx + 1
  idx_after = (idx_after + 1) % 5
  print(t)
  print(averages)
  print(speedup)
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
plt.title("Time vs Threads (" + line.split(' ')[1]+ ")")
plt.legend()
plt.savefig('time' + name)
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
plt.title(" Speedup vs # Threads(" +line.split(' ')[1]+ ")")
plt.legend()
plt.savefig('speedup' + name)
#plt.show()