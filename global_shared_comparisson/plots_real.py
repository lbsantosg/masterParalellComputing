import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
  
def get_plots(t_file,name):
  print(t_file)
  num_blocks = t_file['Num_blocks'].value_counts().keys().sort_values()
  colors = ['red','blue','green','pink','purple','grey'];
  all_plots = {
    "time": {
      "column" : "Average",
      "name"   : name + '_Time_cuda_shared',
      "title"  : "Time in Image Size= {}px".format(name),
      "xlabel" : "# of threads/block",
      "ylabel" : "Seconds"
    },
    "speedup": {
      "column" : "Speedup",
      "name"   : name + '_Speedup_cuda_shared',
      "title"  : "Speedup in Image Size= {}px".format(name),
      "xlabel" : "# of threads/block",
      "ylabel" : "Speedup"
    }
  };

  for image_plot in all_plots: 
    current_plot = all_plots[image_plot];
    pl_name = current_plot["name"]; 
    plt.figure(pl_name); 
    plt.plot([], [], label="Only global", lw=0.9,c='k')
    plt.plot([], [], label="Using shared", c='k',lw=0.9,linestyle='dashed');
    for nBlock in range(0,len(num_blocks)):
      blockValues = t_file.loc[(t_file['Num_blocks'] == num_blocks[nBlock]) & (t_file['Threads'] > 4) & (t_file['Threads'] < 256)]; 
      valuesGlobal = "global{}".format(current_plot['column'])
      valuesShared = "shared{}".format(current_plot['column'])
      plt.plot(blockValues['Threads'],blockValues[valuesGlobal], c=colors[nBlock], lw=0.9,label=("{} Blocks".format(num_blocks[nBlock])));
      plt.plot(blockValues['Threads'],blockValues[valuesShared], c=colors[nBlock], lw=0.9,linestyle='dashed');
    plt.title(current_plot["title"]);
    plt.grid(visible=True, which='major', axis='both');
    plt.xlabel("# of threads/block")
    plt.ylabel(current_plot["ylabel"]);
    plt.legend(loc='upper left')
    plt.savefig('plots_'+ pl_name + '.svg', format='svg');



names = ['720','1080','4k','8k'];
collected_times = []
# for time_file in ['Time_shared_file_0.csv','Time_shared_file_1.csv','Time_shared_file_2.csv', 'Time_shared_file_3.csv']:
#   time_files.append(pd.read_csv(time_file, delimiter = ","))
  
for time_file in range(3,4):
  print(("Reading file... {}").format(time_file));
  collected_data = {
    "global":  pd.read_csv("Time_global_file_{}.csv".format(time_file)),
    "shared":   pd.read_csv("Time_shared_file_{}.csv".format(time_file))
  } 
  collected_times.append(collected_data); 

all_secuential =[[0.696260 ,0.693528 ,0.694040],
             [1.577229 ,1.566189 ,1.564025],
             [6.986456 ,7.044606 ,7.132149],
             [28.564512 ,28.807498 ,29.177079]]
secuential = []; 
for values in all_secuential: 
  av = np.mean(np.mean(values));
  secuential.append(av); 

n_file = 0 
for time_file in collected_times:
  summary_data = time_file["global"].loc[:,["Num_blocks", "Threads"]].copy(); 
  for approach in time_file:
    approach_data = time_file[approach]; 
    avg = []
    s_up = []
    for result in approach_data.index:
      av = np.mean(approach_data.loc[result,['Try_1','Try_2','Try_3']])
      avg.append(av)
      s_up.append((secuential[n_file])/av)
    summary_data["{}Average".format(approach)] = avg 
    summary_data["{}Speedup".format(approach)] = s_up
  get_plots(summary_data, names[n_file]);
  n_file += 1 ;

 
