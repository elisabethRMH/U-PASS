#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:54:43 2022
Updated Dec 10, 2024

@author: Elisabeth Heremans
"""
basepath='...'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.style.use('ggplot') #'nature' can also be option but i get problem with latex
matplotlib.rc('font', **font)
plt.rcParams['figure.figsize']=[10,10]

from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt

length=23
accs=np.zeros((length,6))
kap=np.zeros((length,6))
classwise_sens=np.zeros((length,5,5))
classwise_prec=np.zeros((length,5,5))
j=0
# for i in ['','0001','001','01','02']:#'',,'02' ,'01'
for i in ['','0001','0005','001','002','01']:#'',,'02' ,'01'
    if len(i)==0:
        #dird=basepath+'//results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accchans2_subjnorm_sdtrain42pat/'
        dird=basepath+'//results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_5chans2_subjnorm_sdtrain42pat/'
    else:
        #dird=basepath+'//results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl'+i+'_fzfp2accchans2_subjnorm_sdtrain46pat/'
        dird=basepath+'//results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl'+i+'_5chans2_subjnorm_sdtrain46pat/'
    perf=loadmat(dird+'performance.mat')#performance2_perpat_45
    accs[:,j]=perf['acc'][0][0:length]
    kap[:,j]=perf['kappa'][0][0:length]
    
    
    j+=1

from scipy.stats import ttest_rel
# ttest_rel(accs[:,2],accs[:,0])

table={'Accuracy':[np.mean(accs,0),np.std(accs,0)/np.sqrt(length)],
       'Kappa':[np.mean(kap,0),np.std(kap,0)/np.sqrt(length)],
       'Classwise sensitivity': [np.mean(classwise_sens,0),np.std(classwise_sens,0)/np.sqrt(length)],
       'Classwise precision': [np.mean(classwise_prec,0), np.std(classwise_prec,0)/np.sqrt(length)]}
dfd=np.reshape(np.transpose(np.stack([np.nanmean(classwise_prec,0), np.nanstd(classwise_prec,0)/np.sqrt(length)]),(1,0,2)),(5,10),order='F')
dfd=np.reshape(np.array([np.mean(accs,0),np.std(accs,0)/np.sqrt(length)]),(12,))
dfd=np.reshape(np.array([np.mean(kap,0),np.std(kap,0)/np.sqrt(length)]),(12,))
fig,ax=plt.subplots()
pltt=ax.plot(np.mean(accs,0))
ste=np.std(accs,0)/np.sqrt(length)
x=np.arange(6)
ax.fill_between(x,np.mean(accs,0)+ste,np.mean(accs,0)-ste, facecolor='darksalmon', alpha=0.5)
# ax.set_yticks([0.76,0.77,0.78])
ax.set_yticks([0.72,0.73,0.74,0.75])
# ax.set_yticks([0.82,0.83,0.84])
# ax.set_yticks([0.61,0.62])
# ax.set_yticks([0.640,0.645])
ax.set_xticks(x)
# ax.set_xticklabels(['0','0.1','1','10','20'])# ['0\n(0)','0.1\n(44)','1\n(440)','10\n(4400)','20\n(8800)'])
ax.set_xticklabels(['0','0.1','0.5','1','2','10'])# ['0\n(0)','0.1\n(44)','1\n(440)','10\n(4400)','20\n(8800)'])
ax.set_xlabel('Proportion train samples removed (%)')#lute amount of samples removed)')
ax.set_ylabel('Accuracy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/remove_ambigtrainsamples_arnn_acc_5chans-46pat.pdf',bbox_inches='tight')


# fig,ax=plt.subplots()
# pltt=ax.plot(np.mean(kap,0))
# ste=np.std(kap,0)/np.sqrt(23)
# x=np.arange(5)
# ax.fill_between(x,np.mean(kap,0)+ste,np.mean(kap,0)-ste, facecolor='darksalmon', alpha=0.5)
# ax.set_yticks([0.66,0.67,0.68,0.69])
# # ax.set_yticks([0.61,0.62])
# # ax.set_yticks([0.640,0.645])
# ax.set_xticks(x)
# ax.set_xticklabels(['0','0.1','1','10','20'])# ['0\n(0)','0.1\n(44)','1\n(440)','10\n(4400)','20\n(8800)'])
# ax.set_xlabel('Proportion train samples removed (%)')#lute amount of samples removed)')
# ax.set_ylabel('Accuracy')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/remove_ambigtrainsamples_kap_5chans-45pat.pdf',bbox_inches='tight')

# classwise_sens_norm=classwise_sens-classwise_sens[:,:,0:1]
# fig,ax=plt.subplots()
# pltt=ax.plot(np.mean(classwise_sens_norm,0))
# ste=np.std(classwise_sens_norm,0)/np.sqrt(23)
# x=np.arange(5)
# ax.fill_between(x,np.mean(kap,0)+ste,np.mean(kap,0)-ste, facecolor='darksalmon', alpha=0.5)
# ax.set_yticks([0.66,0.67,0.68,0.69])
# # ax.set_yticks([0.61,0.62])
# # ax.set_yticks([0.640,0.645])
# ax.set_xticks(x)
# ax.set_xticklabels(['0','0.1','1','10','20'])# ['0\n(0)','0.1\n(44)','1\n(440)','10\n(4400)','20\n(8800)'])
# ax.set_xlabel('Proportion train samples removed (%)')#lute amount of samples removed)')
# ax.set_ylabel('Accuracy')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # plt.savefig(basepath+'/220427_confidenceAL/figures_paper/remove_ambigtrainsamples_kap_5chans-45pat.pdf',bbox_inches='tight')
