#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to plot active learning results

Created on Mon Nov 28 13:44:45 2022

@author: Elisabeth Heremans
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
basepath='...'

from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, cohen_kappa_score
sys.path.insert(0, basepath+"/Documents/code/general")
from save_functions import *
from mathhelp import softmax, cross_entropy
## Part 1: version with averaging  per patient

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.style.use('ggplot') #'science','nature' can also be option but i get problem with latex 

matplotlib.rc('font', **font)
plt.rcParams['figure.figsize']=[10,10]

nbfolds=10
nbbasenet=10
nbgroup=3

filename=basepath+"/Documents/code/feature_mapping/sleepuzl_subsets_cross45pat.mat" 
files_folds=loadmat(filename)


#UPASS on PSG with only AL then reject
ordering=np.array([40, 30, 14, 26,  2, 41, 37, 22,  8, 24,  3, 31,  7, 33, 42, 23, 35,
        28, 17, 16, 36, 25, 38,  0, 11,  6, 20, 32, 18, 39, 34, 21, 27, 15,
          1, 10, 43, 29, 19, 13,  5, 12, 44,  4,  9]) #this unlabeled with 5 chans
select_foral=ordering[-18:]
rest_noal=ordering[:-18]

acc={}
wf1={}
kappa={}
acc_per_epoch={}
accunlab_per_epoch={}
conf={}
varepi={}
vartot={}
varal={}
entr_begin={}
entr_end={}
subjs=[]
entr_all={}

training_epoch=1
training_epoch=np.arange(1,11)*training_epoch-1#11
keys=['Epistemic\nuncertainty','Entropy','Total\nuncertainty','Aleatoric\nuncertainty']
for string in ['entr_epi']:#'entr_diags',,'entr_tot','entr_ale']:
    acc[string]=[[] for i in range(nbfolds*nbgroup)]
    wf1[string]=[[] for i in range(nbfolds*nbgroup)]
    kappa[string]=[[] for i in range(nbfolds*nbgroup)]
    varepi[string]=[[] for i in range(nbfolds*nbgroup)]
    vartot[string]=[[] for i in range(nbfolds*nbgroup)]
    entr_begin[string]=[[] for i in range(nbfolds*nbgroup)]
    entr_all[string]=[[] for i in range(nbfolds*nbgroup)]
    entr_end[string]=[[] for i in range(nbfolds*nbgroup)]
    varal[string]=[[] for i in range(nbfolds*nbgroup)]
    acc_per_epoch[string]=[[] for i in range(nbfolds*nbgroup)]
    accunlab_per_epoch[string]=[[] for i in range(nbfolds*nbgroup)]

    dirr=basepath+'/results_confidence/adv_DA/sleepuzl/15patcross/version4_45pat/seqslnet_advDA_AL1x10perc_'+string+'4_sleepuzl5chans_subjnorm_sdtrain1pat/'

    for base_net_nb in range(nbbasenet):
        for fold in range(nbfolds):
            for group in range(nbgroup):

                path=dirr+'total{:d}/n{:d}/group{:d}/'.format(base_net_nb, fold,group)
                # path=dirr+'total{:d}/n{:d}/'.format(base_net_nb, fold)
                # path=dirr+'n{:d}/group{:d}/'.format(base_net_nb, fold)
                if not os.path.isdir(path):
                    continue
                dict2=loadmat(path+'test_ret.mat') #FzFp2
                test_ygt= dict2['ygt']
                test_ygt2=np.concatenate([test_ygt[0],test_ygt[1:,-1]])-1
                #labels or no labels
                labeled=np.load(path+'labeledsamples.npy')
                labeled_bool =np.zeros(test_ygt2.shape)
                labeled_bool[labeled]=1
                labeled_bool=labeled_bool.astype(bool)
                subjs.append(dict2['subjects'][0])
                
                path_=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_5chans2_subjnorm_sdtrain46pat/'
                path_=path_+'total{:d}/'.format(base_net_nb)
                test_ret0=loadmat(path_+'test_ret_45pat.mat')#45pat
                test_subj=test_ret0['subjects']
                subj_idx=test_subj==dict2['subjects'][0]
                if np.sum(subj_idx)==0:
                    subj_idx= test_subj==dict2['subjects'][0] +' '
                test_score=test_ret0['score'][:,subj_idx]
                tmp=test_score
    
                diags =np.array( [np.sum(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])])
                test_yhat0=np.argmax(diags,1)
                acc_per_epoch[string][fold*nbgroup+group].append([accuracy_score(test_ygt2, test_yhat0) ])
                accunlab_per_epoch[string][fold*nbgroup+group].append([accuracy_score(test_ygt2[~labeled_bool], test_yhat0[~labeled_bool]) ])
                _,a2=cross_entropy(tmp[:], tmp[:])
                
                test_score=dict2['score']
                dfd=np.load(path+'score_train_unlab.npy')
                
                tmp=test_score
                diags =np.array( [np.sum(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])])
                test_yhat=np.argmax(diags,1)
        
                _,a=cross_entropy(dfd[:], dfd[:])
                _,vals=cross_entropy(np.mean(dfd[:],0), np.mean(dfd[:],0))
                _,entr_beg=cross_entropy(dfd[0],dfd[0])
                _,entr_en=cross_entropy(dfd[-1],dfd[-1])
        

                dfd2=np.diagonal(dfd[:,:,(test_ygt2-1).astype(int)],axis1=1,axis2=2)
                if not np.any(np.isnan(dfd[:training_epoch[-1]+1])):
                    test_yhat2=np.argmax(dfd,2)[training_epoch]
                    acc_per_epoch[string][fold*nbgroup+group][base_net_nb].extend([accuracy_score(test_ygt2, test_yhat2[i]) for i in range(len(training_epoch))])
                    accunlab_per_epoch[string][fold*nbgroup+group][base_net_nb].extend([accuracy_score(test_ygt2[~labeled_bool], test_yhat2[i][~labeled_bool]) for i in range(len(training_epoch))])
                else:
                    print('error')
    
    
                realvar=np.nanmean(np.nanstd(dfd2,0))
                realconf=np.nanmean(np.nanmean(dfd2,0))
                realvaral=np.nanmean(dfd2*(1-dfd2))
                estvar=vals-np.nanmean(a,0)
                estvaral=np.nanmean(a,0)
                estvartot=vals
                
                varepi[string][fold*nbgroup+group].append(np.mean(estvar))
                varal[string][fold*nbgroup+group].append(np.mean(estvaral))
                vartot[string][fold*nbgroup+group].append(np.mean(estvartot))
                entr_begin[string][fold*nbgroup+group].append(np.mean(a[0]))
                entr_end[string][fold*nbgroup+group].append(np.mean(a[-1]))
                entr_all[string][fold*nbgroup+group].append(np.concatenate([np.array([np.mean(a)]),np.mean(a,1)]))
                
                
                acc[string][fold*nbgroup+group].append([accuracy_score(test_ygt2, test_yhat2[-1]), accuracy_score(test_ygt2[labeled_bool], test_yhat2[-1][labeled_bool]), accuracy_score(test_ygt2[~ labeled_bool], test_yhat2[-1][~ labeled_bool])])
                _, _, wf_, _ =precision_recall_fscore_support(test_ygt2, test_yhat2[-1], average='weighted',labels=[0,1, 2, 3, 4])
                _, _, wf1_, _ =precision_recall_fscore_support(test_ygt2[labeled_bool], test_yhat2[-1][labeled_bool],labels=[0,1, 2, 3, 4], average='weighted')
                _, _, wf2_, _ =precision_recall_fscore_support(test_ygt2[~labeled_bool], test_yhat2[-1][~labeled_bool],labels=[0,1, 2, 3, 4], average='weighted')
                wf1[string][fold*nbgroup+group].append([wf_, wf1_, wf2_])
                kappa[string][fold*nbgroup+group].append([cohen_kappa_score(test_ygt2, test_yhat2[-1]), cohen_kappa_score(test_ygt2[labeled_bool], test_yhat2[-1][labeled_bool]), cohen_kappa_score(test_ygt2[~ labeled_bool], test_yhat2[-1][~ labeled_bool])])
            

varalenew=[varal[string][i] for i in range(len(varal[string])) if len(varal[string][i])>0]
varepinew=[varepi[string][i] for i in range(len(varepi[string])) if len(varepi[string][i])>0]
accnew=[acc[string][i] for i in range(len(acc[string])) if len(acc[string][i])>0]
accunlab_per_epochnew =np.array( [accunlab_per_epoch[string][i] for i in range(len(accunlab_per_epoch[string])) if len(accunlab_per_epoch[string][i])>0])
acc_per_epochnew =np.array( [acc_per_epoch[string][i] for i in range(len(acc_per_epoch[string])) if len(acc_per_epoch[string][i])>0])
plt.plot((np.mean(accunlab_per_epochnew[:,:,-1],1)-np.mean(accunlab_per_epochnew[:,:,0],1))[np.argsort(np.mean(varepinew,1))])

color_cycle  =['#E24A33', '#348ABD', '#A60628','forestgreen', '#188487', '#E24A33']
  
fig, ax = plt.subplots()
i=0
for string in acc.keys():
    ax.plot(np.mean(np.array(acc_per_epoch[string]),0) ,c=color_cycle[i])
    i+=1
ax.legend(keys)
ax.set_ylabel('Accuracy')
x=np.arange(0,21,5)
# ax.set_yticks([0.78,0.79,0.8,0.81,0.82])
ax.set_xticks(x)
ax.set_xticklabels(['0','5','10','15','20'])# ['0\n(0)','5\n(5)','10\n(10)','15','20'])
ax.set_xlabel('Epoch')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/al_5chans_1x10_total10-22.svg')


fig, ax = plt.subplots()
for string in ['entr_epi']: #entr_epi in revised paper!
    ax.plot(np.mean((np.array(acc_per_epoch[string])-np.array(acc_per_epoch[string])[:,:,0:1])[select_foral],(0,1)) ,'#E24A33', alpha=1,linewidth=2)
    ax.plot((np.mean((np.array(acc_per_epoch[string])-np.array(acc_per_epoch[string])[:,:,0:1])[rest_noal],(0,1)) ),'#348ABD', alpha=1, linewidth=2)
    ax.plot(np.transpose(np.mean((np.array(acc_per_epoch[string])-np.array(acc_per_epoch[string])[:,:,0:1])[rest_noal],(1)) ),'#348ABD',alpha=0.3)
    ax.plot(np.transpose(np.mean((np.array(acc_per_epoch[string])-np.array(acc_per_epoch[string])[:,:,0:1])[select_foral],(1)) ),'#E24A33',alpha=0.3)
ax.legend(['High epistemic uncertainty','Rest'])
ax.set_ylabel('Accuracy difference')
x=np.arange(0,11,5)
# ax.set_yticks([0.78,0.79,0.8,0.81,0.82])
ax.set_yticks([-0.02,0,0.02,0.04])
# ax.set_yticklabels(['-0.05','0','0.05'])
ax.set_ylim(-0.03,0.05)
ax.set_xticks(x)
ax.set_xticklabels(['0','5','10'])
ax.set_xlabel('Epoch')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/al_5chans_1x10_total23-entrdiags-45pat.pdf',bbox_inches='tight')


#careful! the acc per epoch, we did not explicitly set the labeled instances to the right ones. so even though x% labeled, this is not directly reflected in the accuracy.
#results for paper 5 chan
np.mean(np.concatenate((np.array(accunlab_per_epoch['entr_diags'])[select_foral][:,:,0], np.array(acc_per_epoch['entr_diags'])[rest_noal][:,:,0])))
np.mean(np.concatenate((np.array(accunlab_per_epoch['entr_diags'])[select_foral][:,:,-1], np.array(acc_per_epoch['entr_diags'])[rest_noal][:,:,0])))
np.mean(np.concatenate((np.array(acc_per_epoch['entr_diags'])[select_foral][:,:,0], np.array(acc_per_epoch['entr_diags'])[rest_noal][:,:,0])))
np.mean(np.concatenate((np.array(acc_per_epoch['entr_diags'])[select_foral][:,:,-1], np.array(acc_per_epoch['entr_diags'])[rest_noal][:,:,0])))

fig, ax = plt.subplots()
for string in ['entr_diags']:
    ax.plot(np.mean((np.array(accunlab_per_epoch[string])-np.array(accunlab_per_epoch[string])[:,:,0:1])[select_foral],(0,1)) ,'#E24A33', alpha=1,linewidth=2)
    ax.plot((np.mean((np.array(accunlab_per_epoch[string])-np.array(accunlab_per_epoch[string])[:,:,0:1])[rest_noal],(0,1)) ),'#348ABD', alpha=1, linewidth=2)
    ax.plot(np.transpose(np.mean((np.array(accunlab_per_epoch[string])-np.array(accunlab_per_epoch[string])[:,:,0:1])[rest_noal],(1)) ),'#348ABD',alpha=0.3)
    ax.plot(np.transpose(np.mean((np.array(accunlab_per_epoch[string])-np.array(accunlab_per_epoch[string])[:,:,0:1])[select_foral],(1)) ),'#E24A33',alpha=0.3)
ax.legend(['High epistemic uncertainty','Rest'])
ax.set_ylabel('Accuracy difference')
x=np.arange(0,11,5)
# ax.set_yticks([0.,0.02,0.05,0.1])
ax.set_ylim(-0.03,0.04)
ax.set_yticks([-0.02,0,0.02,0.04])
ax.set_xticks(x)
ax.set_xticklabels(['0','5','10'])
ax.set_xlabel('Epoch')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/al_5chans_1x10_total23-entrdiags-unlab-45pat-2.pdf',bbox_inches='tight')



np.mean(np.array(acc_per_epoch[string])[:,:,0],0)
np.mean(np.concatenate((np.array(acc_per_epoch[string])[select_foral,:,-1], np.array(acc_per_epoch[string])[rest_noal,:,0])),1)
np.mean(np.concatenate((np.array(accunlab_per_epoch[string])[select_foral,:,-1], np.array(acc_per_epoch[string])[rest_noal,:,0])),0) #this one is used as final acc in paper: 

    
fig, ax = plt.subplots()
for string in ['entr_epi']:
    ax.plot(np.mean((np.array(entr_all[string])-np.array(entr_all[string])[:,:,0:1])[select_foral],(0,1)) ,'#E24A33', alpha=1,linewidth=2)
    ax.plot((np.mean((np.array(entr_all[string])-np.array(entr_all[string])[:,:,0:1])[rest_noal],(0,1)) ),'#348ABD', alpha=1, linewidth=2)
    ax.plot(np.transpose(np.mean((np.array(entr_all[string])-np.array(entr_all[string])[:,:,0:1])[rest_noal],(1)) ),'#348ABD',alpha=0.3)
    ax.plot(np.transpose(np.mean((np.array(entr_all[string])-np.array(entr_all[string])[:,:,0:1])[select_foral],(1)) ),'#E24A33', alpha=0.3)
ax.legend(['High epistemic uncertainty','Rest'])
ax.set_ylabel('Entropy difference')
x=np.arange(0,11,5)
# ax.set_yticks([0.78,0.79,0.8,0.81,0.82])
ax.set_xticks(x)
ax.set_xticklabels(['0','5','10'])# ['0\n(0)','5\n(5)','10\n(10)','15','20'])
ax.set_xlabel('Epoch')#'\n(Proportion labeled samples in %)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig, ax = plt.subplots()
for string in ['entr_epi']:
    accs_a=([np.mean(np.array(acc[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    accs1_a=([np.mean(np.array(accunlab_per_epoch[string][i])[:,0],0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    accsend_a=([np.mean(np.array(accunlab_per_epoch[string][i])[:,-1],0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    varepi_a=([np.mean(np.array(varepi[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    varal_a=([np.mean(np.array(varal[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    vartot_a=([np.mean(np.array(vartot[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    entrbeg_a=([np.mean(np.array(entr_begin[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
    entrend_a=([np.mean(np.array(entr_end[string][i]),0) for i in range(len(acc[string])) if len(acc[string][i])>0] )
np.concatenate(np.concatenate(files_folds['test_sub'][0]))[np.argsort(np.array(accs_a)[:,0])]
np.concatenate(np.concatenate(files_folds['test_sub'][0]))[np.argsort(np.array(varepi_a)[:])]
np.concatenate(np.concatenate(files_folds['test_sub'][0]))[np.argsort(np.array(varal_a)[:])]
np.array(accs_a)[:,0][np.argsort(np.array(varepi_a)[:])]
accsdiff=np.array(accsend_a)-np.array(accs1_a)
plt.plot(accsdiff[np.argsort(np.array(vartot_a)[:])])
plt.plot(accsdiff[np.argsort(np.array(varepi_a)[:])])

from scipy.stats.stats import pearsonr
pearsonr(accsdiff, (np.array(varepi_a)[:]))

#Based on unsupervised AL, we define the ordering of patients (least epistemic uncertainty to most), and we select the most epistemic uncertain ones for AL.
#for 5 chans PSG
subjects=np.concatenate(np.concatenate(files_folds['test_sub'][0]))
keep_subjs_bool=(subjects!=55)*(subjects!=66)*(subjects!=57)
ordering=np.array([40, 30, 14, 26,  2, 41, 37, 22,  8, 24,  3, 31,  7, 33, 42, 23, 35,
        28, 17, 16, 36, 25, 38,  0, 11,  6, 20, 32, 18, 39, 34, 21, 27, 15,
         1, 10, 43, 29, 19, 13,  5, 12, 44,  4,  9]) #obv unlabeled with 5 chans
select_foral=ordering[-18:]
rest_noal=ordering[:-18]
accsdiff[ordering[-5:]][keep_subjs_bool[ordering[-5:]]]
np.concatenate((np.array(accs1_a)[ordering[:-5]][keep_subjs_bool[ordering[:-5]]],
np.array(accsend_a)[ordering[-5:]][keep_subjs_bool[ordering[-5:]]]))


fig, ax = plt.subplots()
ax.boxplot([np.array(accunlab_per_epoch[string])[:,-1]-np.array(accunlab_per_epoch[string])[:,0] for string in acc.keys() ])
ax.set_ylabel('Accuracy increase')
ax.set_xticklabels(keys)
# ax.set_xlabel('Sleep effiency')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([-0.1,0,0.1,0.2])
# plt.boxplot([acc[:,0],acc2[:,0],acc3[:,0]])
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/al_2chans_1x10_total10_accinc-2.svg')

from scipy import stats
stats.wilcoxon(np.array(acc['entr_diags'])[:,0], np.array(acc['entr_ale'])[:,0])

