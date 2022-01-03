from matplotlib import pyplot as plt
import numpy as np
import re

plt.rcParams.update({'font.size': 18})

def get_log_data(file_name):
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    file1.close()

    frequency_lists=[]
    psnr_list=[]
    ratio_list=[]

    for line in Lines:
        strings = re.split(':|,', line.strip())
        #print (strings)
        fre_list = []
        fre_list.append(float(strings[-7][2:]))
        fre_list.append(float(strings[-6]))
        fre_list.append(float(strings[-5]))
        fre_list.append(float(strings[-4]))
        fre_list.append(float(strings[-3][:-1]))
        frequency_lists.append(np.array(fre_list))

        psnr_list.append(float(strings[5]))
        ratio_list.append(float(strings[-1]))


    return frequency_lists, np.array(psnr_list), np.array(ratio_list)

def get_fbc_fig(all_norms,num_iter,ylim=1,save_path='',img_name=''):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlim(0,num_iter)
    ax.set_ylim(0, ylim)

    norms=np.array(all_norms)

    label_list = ['Frequency band (1,lowest)','Frequency band (2)','Frequency band (3)','Frequency band (4)','Frequency band (5, highest)']

    plt.xlabel("Optimization Iteration")
    plt.ylabel("FBC ($\\bar{H}$)")
    #plt.title('FBC (%s)'%img_name)

    color_list = ['#331900', '#994C00', '#CC6600',  '#FF8000', '#FF9933']
    rate = 1
    for i in range(norms.shape[1]):
        plt.plot(range(0,num_iter,rate), norms[:num_iter:rate,i], linewidth=4, color=color_list[i], label=label_list[i]) 

    plt.legend(loc=4,)
    plt.grid()
    plt.savefig(save_path)
    #plt.show()
    plt.close()


def get_psnr_ratio_fig(all_datas,num_iter,ylim=35, ylabel='',save_path='',img_name=''):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlim(0,num_iter)
    ax.set_ylim(0, ylim)

    plt.xlabel("Optimization Iteration")
    #plt.ylabel(ylabel)
    #plt.title(img_name)

    label_list = ['PSNR','Ratio']
    color_list = ['#d94a31','#4b43db']

    rate = 1
    for i in range(len(all_datas)):
        plt.plot(range(0,num_iter,rate), all_datas[i][0:num_iter:rate], linewidth=4, color=color_list[i], label=label_list[i])

    plt.legend(loc=0,)
    plt.grid()
    plt.savefig(save_path)
    #plt.show()
    plt.close()