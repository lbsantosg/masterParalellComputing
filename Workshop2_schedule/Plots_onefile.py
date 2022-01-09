import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
name =  "_plot.png"
data = []
pictures = ['720p', '1080p', '4k', '8k']
npics = 4
for i in range(npics): 
  data.append(pd.read_csv("Time_file_"+str(i)+".csv", delimiter = ","))
for i in range(npics):
  averages = []
  speedup = []
  secuential = 1 
  for t in data[i].index:
    av = 0 
    for tries in data[i][['Try_1','Try_2','Try_3']]:
      av += data[i].loc[t,tries]
    averages.append((av/3.0))
    if (data[i].loc[t]['Threads'] == 1.0):
      secuential = av/3.0
    speedup.append(secuential/(av/3.0))
  data[i]["Average"] = averages
  data[i]["Speedup"] = speedup
  print(data[i])
for i in range(npics):
  plt.plot(data[i]['Threads'],data[i]['Average'], label = pictures[i])
plt.xlabel("Threads")
plt.ylabel("Seconds")
plt.title("Time vs Threads using OMP")
plt.legend()
plt.savefig('schedule_time' + name)
#plt.show()
plt.close()
for i in range(npics):
  plt.plot(data[i]['Threads'],data[i]['Speedup'], label = pictures[i])
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads using OMP")
plt.legend()
plt.savefig('schedule_speedup' + name)
#plt.show()