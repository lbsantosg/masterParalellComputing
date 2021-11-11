import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files 
import numpy as np

names = ['720','1080','4k']
time_files = []
for time_file in ['Time__file_0.csv','Time__file_1.csv','Time__file_2.csv']:
  time_files.append(pd.read_csv(time_file, delimiter = ","))
  
secuential =[[0.475591,1.200146,2.103172,3.330360,5.093617,8.119533,9.855713],
             [1.210031,2.889327,5.090738,7.821800,11.588192,16.930748,21.710766],
             [3.284441,8.489567,15.871103,25.710022,37.829060,56.907367,74.501744]]
n_file = 0 
for time_file in time_files: 
  avg = []
  s_up = []
  #print(n_file)
  for result in time_file.index:
    av = np.mean(time_file.loc[result,['Try_1','Try_2','Try_3']])
    avg.append(av)
    s_up.append((secuential[n_file][result%7])/av)
  time_file["Average"] = avg 
  time_file["Speedup"] = s_up
  n_file += 1 
  #print(time_file)
  
  def get_plots(t_file,name):
  fz = 12

  num_blocks = t_file['Num_blocks'].value_counts().keys().sort_values()
  kernel_sz = t_file['Kernel_size'].value_counts().keys().sort_values()
  colors = ['red','blue','green','pink','purple','yellow','grey']
  for n_block in range(len(num_blocks)):
    block = t_file.loc[t_file['Num_blocks'] == num_blocks[n_block]]
    plt.rcParams.update({'font.size': fz})

    for kernel in range(len(kernel_sz)):
      k = block.loc[block['Kernel_size'] == kernel_sz[kernel]]
      plt.plot(k['Threads'],k['Average'], color=colors[kernel],label=('Kernel {}'.format(kernel_sz[kernel])))
   
  
    plt.title('Number of blocks= {}'.format(num_blocks[n_block]))
    plt.xlabel("# of threads/block")
    plt.ylabel("Seconds")
    plt.legend()
    pl_name = name + '_Time_cuda_'  + str(num_blocks[n_block]) + '_blocks.png'
    plt.savefig(pl_name)
    files.download(pl_name)
    plt.show()
    plt.rcParams.update({'font.size': fz})

    for kernel in range(len(kernel_sz)):
      k = block.loc[block['Kernel_size'] == kernel_sz[kernel]]
      plt.plot(k['Threads'],k['Speedup'], color=colors[kernel],label=('Kernel {}'.format(kernel_sz[kernel])))

    plt.title('Number of blocks= {}'.format(num_blocks[n_block]))
    plt.xlabel("# of threads/block")
    plt.ylabel("Speedup")
    plt.legend()
    pl_name = name + '_Speedup_cuda_'  +str(num_blocks[n_block]) + '_blocks.png'
    plt.legend()
    plt.savefig(pl_name)
    files.download(pl_name)
    plt.show()
 
