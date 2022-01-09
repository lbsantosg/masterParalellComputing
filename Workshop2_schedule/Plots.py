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

plt.plot(data['Threads'],data['Average'], color='blue')
plt.xlabel("Threads")
plt.ylabel("Seconds")
plt.title("Time vs Threads (" + line.split(' ')[1]+ ") using OMP")
plt.legend()
plt.savefig('schedule_time' + name)
#plt.show()
plt.close()
plt.plot(data['Threads'],data['Speedup'], color='blue')
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads (" +line.split(' ')[1]+ ") using OMP")
plt.legend()
plt.savefig('schedule_speedup' + name)
#plt.show()