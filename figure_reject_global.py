#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure to show the 
Created on Thu Mar  2 17:51:19 2023

@author: Elisabeth Heremans
"""
#Plots
import sys
basepath='...'
sys.path.insert(0,basepath+"/Documents/code/general")


from mathhelp import softmax, cross_entropy
from uncertainty_metrics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scienceplots
from sklearn.metrics import accuracy_score
from scipy.io import loadmat, savemat
import os

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.style.use('ggplot') #'nature' can also be option but i get problem with latex
matplotlib.rc('font', **font)
plt.rcParams['figure.figsize']=[10,10]


# # matplotlib.rc('font', **font)
filename2=basepath+"/Documents/code/feature_mapping/sleepuzl_subsets_cross45pat.mat" #Fz or Fp2, they are in channel 5
files_folds2=loadmat(filename2)
subjects=np.concatenate(np.concatenate(files_folds2['test_sub'][0]))


#UPASS on PSG with only AL then reject
volgorde=np.array([40, 30, 14, 26,  2, 41, 37, 22,  8, 24,  3, 31,  7, 33, 42, 23, 35,
        28, 17, 16, 36, 25, 38,  0, 11,  6, 20, 32, 18, 39, 34, 21, 27, 15,
          1, 10, 43, 29, 19, 13,  5, 12, 44,  4,  9]) #obv unlabeled with 5 chans
select_foral=volgorde[-18:]
rest_noal=volgorde[:-18]



#UPASS on wearable again with only reject
#So on wearable we don;t use the 'volgorde' and active learning thing. we only do reject. also some patients rejected beforehand (deleted_subjects)
#on PSG we do both active learning (using volgorde, 18 last patients) and reject samples. but no reject full patients (deleted subjects is empty)

# accsend_a[volgorde[-7:]][keep_subjs_bool[volgorde[-7:]]]

def uncertainty_metric(distances, confidences, alpha=1):
    """
    Calculate an uncertainty metric based on distances and confidences.

    Parameters:
    distances (numpy.ndarray): A 2D array where each row represents the distances of a sample to all training samples.
    confidences (numpy.ndarray): A 2D array where each row represents the confidence scores of all training samples.
    alpha (float, optional): A scaling factor for the exponential function. Default is 1.

    Returns:
    numpy.ndarray: A 1D array where each element represents the uncertainty metric for a corresponding sample.
    """
    ''''''
    dfd=softmax(distances)*confidences
    dfd=np.sum(dfd,1)
    fac=np.exp(alpha*np.mean(distances,1))
    return dfd*fac

def calculate_personalization_values(out_dir2):
    """
    Calculate personalization values based on the provided directory.
    Args:
        out_dir2 (str): The directory containing the necessary files for calculation.
    Returns:
        tuple: A tuple containing:
            - vals00 (numpy.ndarray): Calculated values based on the key.
            - diags00 (numpy.ndarray): Diagonal values from the score matrix.
            - diags01 (numpy.ndarray): Mean diagonal values from the ground truth matrix.
            - labeled_samples (numpy.ndarray): Array of labeled sample indices.
    """
    extra_str='_DIAGS_45'
    # dict4=loadmat(os.path.join(out_dir2, "test_ret.mat"))
    dict4=loadmat(os.path.join(out_dir2, "test_retFzFp2.mat"))
    
    valsconf=np.load(out_dir2+'/vals_conf'+extra_str+'.npy',allow_pickle=True).item()
    args_select= np.argwhere(valsconf['dist closest easy training']<3* np.median(valsconf['dist closest easy training']))

    yyhateog1=dict4['yhat']
    scoreeog1=dict4['score']
    ygt1=dict4['ygt']
    sorted1=np.load(out_dir2+'/sorted'+extra_str+'.npy',allow_pickle=True)
    sorted1=sorted1.item()
    valsconf['Max probability']=[]

    labeled_samples=np.load(os.path.join(out_dir2, 'labeledsamples.npy'))
    labeled_bool =np.zeros(valsconf['entropy'].shape[0])
    labeled_bool[labeled_samples]=1
    labeled_bool=labeled_bool.astype(bool)                                    

    valsconftmp=valsconf[key]
    if len(np.array(valsconftmp).shape)>1:
        valsconftmp=np.mean(valsconftmp,1)
    if 'conf' in key or 'perf' in key:
        valsconftmp=-valsconftmp
    vals00=valsconftmp
 
    tmp1=scoreeog1
    tmp2=ygt1
    diags =np.array( [np.sum(np.diagonal(np.flip(tmp1,0),i),1) for i in range(-tmp1.shape[0]+1,tmp1.shape[1])])
    diags00=np.argmax(diags,1)
    diags = np.array([np.mean(np.mean(np.diagonal(np.flip(tmp2,0),i))) for i in range(-tmp2.shape[0]+1,tmp2.shape[1])])
    diags01=np.array(diags)-1

    if key=='Max probability':
        tmp=scoreeog1
        diags = [np.mean(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])]
        diags=np.array(diags)
        vals00=1-np.max(diags,1)#vals0
    elif 'closest training' in key:
        dists=valsconf['dist closest']
        valss=valsconf[key]
        if 'conf' in key or 'perf' in key:
            valss=1-valsconf[key]
        dfd=uncertainty_metric(dists, valss,alpha=10)
        vals00=dfd
    
    return vals00, diags00, diags01, labeled_samples

filter_sorted=False
dict2={}
dict2_noslstch={}
dict2_percentslstch={}
value_on13perc=[]
howmuchbelowthresh=[]
extra_str='_DIAGS'

for fold in range(10):
    labeled=[]
    for pat_group in range(1):
        dir2= '/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_5chans2_subjnorm_sdtrain46pat'
        
        #out_dir2 = os.path.join(dir2,'n{:d}/group0'.format(fold))
        out_dir2 = os.path.join(dir2,'total{:d}'.format(fold))
        dict3=loadmat(os.path.join(out_dir2, "test_retSD2.mat"))

        #Load the confidence values of the training samples.
        valsconf=np.load(out_dir2+'/vals_conf'+extra_str+'.npy',allow_pickle=True).item()
        #deleted_subjects=[56, 57, 54, 77, 76, 99, 74] #only for wearable (U-PASS paper 2)
        deleted_subjects=[]
        args_remove= []
        args_select = np.arange(len(dict3['subjects2']))
        yyhateog1=dict3['yhat']
        scoreeog1=dict3['score']
        ygt1=dict3['ygt']
        sorted1=np.load(out_dir2+'/sorted'+extra_str+'.npy',allow_pickle=True)
        sorted1=sorted1.item()
        valsconf['Max probability']=[]

        
        if len(dict2)==0:
            for key in valsconf.keys():
                dict2[key]=[]
                dict2_noslstch[key]=[]
                dict2_percentslstch[key]=[]
        keyNb=0

        #Try different uncertainty measures
        for key in valsconf.keys():


            valsconftmp=valsconf[key]
            if len(np.array(valsconftmp).shape)>1:
                valsconftmp=np.mean(valsconftmp,1)
            if 'conf' in key or 'perf' in key:
                valsconftmp=-valsconftmp
            vals00=valsconftmp
            sleepstagechanges=[]
            if key=='Max probability':
                vals00=np.zeros(( valsconf['entropy'].shape[0]))
            
            if filter_sorted:
                sortedtmp= np.array([i for i in sortedtmp if i in args_select])

            subjs=np.unique(dict3['subjects'])
            diags00=np.zeros((valsconf['entropy'].shape[0]))
            diags01=np.zeros((valsconf['entropy'].shape[0]))

            #Go recording per recording
            for s in subjs:
                
                tmp1=scoreeog1[:,dict3['subjects']==s]
                tmp2=ygt1[:,dict3['subjects']==s]  
                

                diags =np.array( [np.sum(np.diagonal(np.flip(tmp1,0),i),1) for i in range(-tmp1.shape[0]+1,tmp1.shape[1])])
                diags00[dict3['subjects2']==s]=np.argmax(diags,1)
                diags = np.array([np.mean(np.mean(np.diagonal(np.flip(tmp2,0),i))) for i in range(-tmp2.shape[0]+1,tmp2.shape[1])])
                diags01[dict3['subjects2']==s]=np.array(diags)-1

                if key=='Max probability':
                    tmp=scoreeog1[:,dict3['subjects']==s]  
                    diags = [np.mean(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])]
                    diags=np.array(diags)
                    vals00[dict3['subjects2']==s]=1-np.max(diags,1)#vals0
            
            
            if key =='conf closest training' or key =='varale closest training' or key =='var closest training' or key=='perf closest training':#key =='Mix':
                dists=valsconf['dist closest']
                valss=valsconf[key]
                alpha=10
                if 'conf' in key or 'perf' in key:
                    valss=1-valsconf[key]
                    alpha=10
                dfd=uncertainty_metric(dists, valss,alpha=alpha)
                vals00=dfd
            
            for s in subjs:
                subject=int(s[4:])
                if subject in subjects[select_foral]:
                    #dir2='/volume1/scratch/ehereman/results_confidence/adv_DA/sleepuzl/15patcross/version4_45pat/seqslnet_advDA_AL1x10perc_entr_diags4_sleepuzlfzfp2accchans_subjnorm_sdtrain1pat/'      
                    dir2='/volume1/scratch/ehereman/results_confidence/adv_DA/sleepuzl/15patcross/version4_45pat/seqslnet_advDA_AL1x10perc_entr_epi4_sleepuzl5chans_subjnorm_sdtrain1pat/'      

                    findfold=np.where([np.sum(files_folds2['test_sub'][0][i]==subject) for i in range(len(files_folds2['test_sub'][0]))])[0][0]
                    findgroup=np.where((files_folds2['test_sub'][0][findfold]==subject)[0])[0][0]
                    
                    out_dir2 = os.path.join(dir2,'total{:d}/n{:d}/group{:d}'.format(fold,findfold,findgroup))#{:d}'.format(fold,pat_group))
                    
                    values_tmp, yh_tmp, ygt_tmp, labeled= calculate_personalization_values(out_dir2)
                    vals00[dict3['subjects2']==s]=values_tmp
                    diags01[dict3['subjects2']==s]=ygt_tmp
                    diags00[dict3['subjects2']==s]=yh_tmp
                    
                    args_remove.extend(list( np.where(dict3['subjects2']==s)[0][0]+labeled))
            
            
            usedlabels=diags01
            tmp=usedlabels[1:]-usedlabels[0:-1]
            slstch=np.zeros(len(usedlabels),dtype=bool)
            slstch[1:]=tmp!=0
            slstch2=np.zeros(len(usedlabels),dtype=bool)
            slstch2[:-1]=tmp!=0
            slstch3=slstch+slstch2
            sleepstagechanges=slstch3

            sortedtmp=np.flipud(np.argsort(vals00))#np.flipud(
            
            
            ##FOr dreem:
            #thresh=np.array([0.87582963, 0.84667816, 0.08216709, 0.97994577, 0.34305093, 0.36251731, 0.61890958, 0.07994215, 3.41349416])
            
            sortedtmp= np.flipud(sortedtmp) #gesorteerd met toenemende onzekerheid (dus eerste x nemen is meest zekere)
            sortedtmp= np.array([i for i in sortedtmp if i in args_select and i not in args_remove])
            value_on13perc.append(vals00[sortedtmp][int(len(sortedtmp)*(1-0.15))])
            #howmuchbelowthresh.append(np.sum(vals00<thresh[keyNb]))
            argsort_yyhateog4=diags00[sortedtmp]
            argsort_ygt4=diags01[sortedtmp]
            sleepstagechanges=sleepstagechanges[sortedtmp]
            
                
            acc_argsort4=[]
            acc_noslstch_argsort4=[]
            percent_slstch=[]
            for i in np.arange(0.01*len(argsort_ygt4),len(argsort_ygt4)+1,len(argsort_ygt4)/100):
                i=int(i)
                acc_argsort4.append(accuracy_score(argsort_ygt4[:i].flatten().astype(int),argsort_yyhateog4[:i].flatten().astype(int)))
                acc_noslstch_argsort4.append(accuracy_score(argsort_ygt4[:i][~sleepstagechanges[:i]].flatten().astype(int),argsort_yyhateog4[:i][~sleepstagechanges[:i]].flatten().astype(int)))
                percent_slstch.append(np.mean(sleepstagechanges[:i]))
            dict2[key].append(acc_argsort4)
            dict2_noslstch[key].append(acc_noslstch_argsort4)
            dict2_percentslstch[key].append(percent_slstch)
            keyNb+=1
    

#Figure 3c

keys=['entropy',
 'distance class proportion',
 'dist closest easy training',
 'conf closest training',
 'varale closest training',
 'varepi closest training',
 'Max probability']
# keys=['entropy',
#   'conf closest training',
#   'varale closest training']# keys.extend( list(dict2.keys())[7:])
# keys=['conf closest training']
y=np.arange(0.01,1.001,0.01)*100
keys2=['Entropy',
 'Proportion tr. class distances',
 'Distance tr.',
 'Weighted confidence closest tr.',
 'Weighted aleatoric uncertainty closest tr.',
 'Weighted epistemic uncertainty closest tr.',
 'Max probability']
# keys2=['Weighted confidence closest tr.']
color_cycle  =[ '#348ABD','#E24A33', '#A60628','forestgreen', '#22b1b5', '#DB9194', '#777777', '#F7B74A']

# keys2=['Entropy',
#   'Weighted confidence closest tr.',
# 'Weighted aleatoric uncertainty closest tr.']#,'Mix']

fig, ax = plt.subplots()
datameans=np.zeros((100,len(keys)))
i=0
for key in keys:          
    data=np.stack(dict2[key],axis=-1)
    datamean=np.mean(data,1)
    # ax.plot(y,data,c=color_cycle[i],alpha=0.2)

    # ax.plot(y,datamean, linewidth=2)
    ax.plot(y,datamean, linewidth=3,c=color_cycle[i])
    datameans[:,i]=datamean
    i+=1
ax.legend(keys2, fontsize=16)

ax.set_yticks([0.8,0.9,1])
# ax.set_yticks([0.4,0.5,0.6,0.7,0.8])
# ax.set_yticks([0.3,0.4,0.5,0.6])
ax.set_xlabel('% most uncertain test data')
ax.set_ylabel('Accuracy')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/reject_sd22chan_noAL_diags_23folds22-inv-45.pdf')
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/reject_5chan_afterAL_diags_23folds2-45.pdf',bbox_inches='tight')

#the best method for every data regime
minkeys= np.array(keys)[np.argmax(datameans,1)]
ys=[y[minkeys==key] for key in keys]
[np.argmin(datameans[datameans[:,i]>0.75,i],0) for i in range(datameans.shape[1])]
seq=[np.argmin(datameans[datameans[:,i]>0.7,i],0) for i in range(datameans.shape[1])]
[datameans[seq[i],i] for i in range(len(seq))]
# in 5 chans, 1730/42551 = 4.07% labeled
#0.21*40821/42551 = 20%

100-y[seq]

#6%*(38/45*98.5/100)=5%
#94*(38/45*98.5/100)=78%

#FOR dreem osa: 90% if 13% thrown away. not sure yet how much percen tthat is of the total! 25950/26990