import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
  
def get_plots(t_file,name):
  num_blocks = t_file['Num_blocks'].value_counts().keys().sort_values()
  colors = ['red','blue','green','pink','purple','grey'];

  all_plots = {
    "time": {
      "column" : "Average",
      "name"   : name + '_Time_cuda_global',
      "title"  : "Time in Image Size= {}px".format(name),
      "xlabel" : "# of threads/block",
      "ylabel" : "Seconds"
    },
    "speedup": {
      "column" : "Speedup",
      "name"   : name + '_Speedup_cuda_global',
      "title"  : "Speedup in Image Size= {}px".format(name),
      "xlabel" : "# of threads/block",
      "ylabel" : "Speedup"
    }
  };

  for image_plot in all_plots: 
    current_plot = all_plots[image_plot];
    pl_name = current_plot["name"]; 
    plt.figure(pl_name); 
    for nBlock in range(0,len(num_blocks)):
      blockValues = t_file.loc[(t_file['Num_blocks'] == num_blocks[nBlock]) & (t_file['Threads'] > 4) & (t_file['Threads'] < 256)]; 
      plt.plot(blockValues['Threads'],blockValues[current_plot['column']], color=colors[nBlock], label=("{} Blocks".format(num_blocks[nBlock])))
    plt.title(current_plot["title"]);
    plt.grid(visible=True, which='major', axis='both');
    plt.xlabel("# of threads/block")
    plt.ylabel(current_plot["ylabel"]);
    plt.legend()
    plt.savefig('plots_'+ pl_name + '.svg', format='svg');



names = ['720','1080','4k','8k'];
time_files = []
for time_file in ['Time_global_file_0.csv','Time_global_file_1.csv','Time_global_file_2.csv', 'Time_global_file_3.csv']:
  time_files.append(pd.read_csv(time_file, delimiter = ","))
  

all_secuential =[[0.696260 ,0.693528 ,0.694040],
             [1.577229 ,1.566189 ,1.564025],
             [6.986456 ,7.044606 ,7.132149],
             [28.564512 ,28.807498 ,29.177079]]
secuential = []; 
for values in all_secuential: 
  av = np.mean(np.mean(values));
  secuential.append(av); 

n_file = 0 
for time_file in time_files: 
  avg = []
  s_up = []
  #print(n_file)
  for result in time_file.index:
    av = np.mean(time_file.loc[result,['Try_1','Try_2','Try_3']])
    avg.append(av)
    s_up.append((secuential[n_file])/av)
  time_file["Average"] = avg 
  time_file["Speedup"] = s_up
  get_plots(time_file, names[n_file]);
  n_file += 1 

 
