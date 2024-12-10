#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SeqSleepNet trained with confidence estimates on the test data,
then compute distances from test samples to training samples and compute
test set uncertainty metrics.

Also code to make figures about this all.
Figure 2 (training dynamics & UMAPS)
Figure 3a (training bananas/ dataset cartography)
Created on Thu Oct 29 17:01:10 2020

@author: Elisabeth Heremans
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
import scipy
import umap
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat

#from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap
import matplotlib.pyplot as plt
import math
sys.path.insert(1,basepath+'/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet') #22/05: changed SeqSleepNet_E into SeqSleepNet
from seqsleepnet_sleep import SeqSleepNet_Sleep
sys.path.insert(1,basepath+'/Documents/code/confidence_AL/')
from seqsleepnet_sleep_withconf import SeqSleepNet_Sleep_Conf

from seqsleepnet_sleep_config import Config
sys.path.insert(1,basepath+'/Documents/code/feature_mapping/')
sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *
from mathhelp import softmax, cross_entropy

from distribution_comparison import distribution_differences

sys.path.insert(0, "/users/sista/ehereman/Documents/code/adapted_tb_classes")
from subgenfromfile_epochsave import SubGenFromFile

filename="/users/sista/ehereman/Documents/code/feature_mapping/sleepuzl_subsets_46pat.mat" #Fz or Fp2, they are in channel 5
filename2="/users/sista/ehereman/Documents/code/feature_mapping/sleepuzl_subsets_cross45pat.mat" #Fz or Fp2, they are in channel 5

files_folds=loadmat(filename)
files_folds2=loadmat(filename2)

normalize=True

source='/esat/stadiustempdatasets/ehereman/SleepData/processed_tb/allfiles-FzFp2sd-moreeegchan2/'
source2='/esat/stadiustempdatasets/ehereman/SleepData/processed_tb/allfiles-FzFp2sd-moreeegchan2/' #in paper it is first69files but i threw this away

number_patients=2
#VERSION WITH PATIENT GROUPS
dim=15*int(26/number_patients)
acc_matrix=np.zeros((dim,4))
kap_matrix=np.zeros((dim,4))
acc_matrixC=np.zeros((dim,4))
kap_matrixC=np.zeros((dim,4))
ind=0
for fold in range(23):
    for pat_group in range(1):#int(26/number_patients)):
        
    
        eval_files=files_folds['eval_sub'][0][fold][0]
        train_files=files_folds['train_sub'][0][fold][0]

        test_files=np.concatenate([files_folds2['test_sub'][0][0][0], files_folds2['eval_sub'][0][0][0],files_folds2['train_sub'][0][0][0]])
        

        config= Config()
        config.epoch_seq_len=10
        config.epoch_step=config.epoch_seq_len
        
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True,file_per_subject=True)
    
        train_generator= SubGenFromFile(source2,shuffle=False, batch_size=config.batch_size,subjects_list=train_files,  sequence_size=config.epoch_seq_len,normalize_per_subject=True,file_per_subject=True)
    
        # eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: ././{:d}".format(  len(test_generator.datalist)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: ./{:d}".format( test_batches_per_epoch))
        config.out_dir1= '/esat/asterie1/scratch/ehereman/results_SeqSleepNet_tb/totalmass2/seqsleepnet_sleep_nfilter32_seq10_dropout0.75_nhidden64_att64_1chan_subjnorm/total'.format( fold+1)

        #dir2= '/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_5chans2_subjnorm_sdtrain42pat'
        #dir2='/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_fzfp2accchans2_subjnorm_sdtrain46pat/total{:d}'.format(fold)
        dir2='/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_5chans2_subjnorm_sdtrain46pat/total{:d}'.format(fold)        # dir2=basepath+'/results_confidence/adv_DA/sleepuzl/15patcross/version4/seqslnet_advDA_AL1x10perc_entr_epi4_sleepuzl5chans_subjnorm_sdtrain1pat/'
        config.out_dir2 = dir2# os.path.join(dir2,'total{:d}'.format(fold))#{:d}'.format(fold,pat_group))

        config.checkpoint_dir= './checkpoint/'
        config.allow_soft_placement=True
        config.log_device_placement=False
        config.nchannel=5
        config.out_path1= config.out_dir1
        config.out_path = config.out_dir2
        config.epoch_step=config.epoch_seq_len
        config.learning_rate= 1E-4
        config.training_length=18000#9000
        config.channel = 0
        config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_path1,config.checkpoint_dir))
        assert(os.path.isdir(os.path.abspath(config.out_path1)))
        assert(os.path.isdir(os.path.abspath(config.checkpoint_path1)))
        config.checkpoint_path = os.path.abspath(os.path.join(config.out_dir2,config.checkpoint_dir))
            

        
        with tf.Graph().as_default() as tl_graph:
            session_conf2 = tf.ConfigProto(
              allow_soft_placement=config.allow_soft_placement,
              log_device_placement=config.log_device_placement)
            session_conf2.gpu_options.allow_growth = True
            sess2 = tf.Session(graph=tl_graph, config=session_conf2)
            with sess2.as_default():
                net = SeqSleepNet_Sleep(config=config) #make seqsleepnet network
                
                # for batch normalization
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Define Training procedure
                    global_step2 = tf.Variable(0, name="global_step", trainable=False)
                    optimizer2 = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
                    grads_and_vars2 = optimizer2.compute_gradients(net.loss)
                    train_op2 = optimizer2.apply_gradients(grads_and_vars2, global_step=global_step2)
                    
        
                print("Writing to {}\n".format(config.out_dir2))
        
                saver2 = tf.compat.v1.train.Saver(tf.all_variables(), max_to_keep=1)
        
        
                # Restore all variables
                print("Model initialized")
                best_dir2 = os.path.join(config.checkpoint_path, 'best_model_acc')#"best_model_acc")
                saver2.restore(sess2, best_dir2)
                print("Model loaded")
        
        
                def dev_step(x_batch, y_batch):
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                        net.input_x: x_batch,
                        net.input_y: y_batch,
                        net.dropout_keep_prob_rnn: 1.0,
                        net.epoch_seq_len: epoch_seq_len,
                        net.frame_seq_len: frame_seq_len,
                        net.istraining: 0
                    }
                    output_loss, total_loss, yhat, score, features = sess2.run(
                           [net.output_loss, net.loss, net.predictions, net.scores, net.rnn_out2], feed_dict)
                    return output_loss, total_loss, yhat, score, features



                def evaluate(gen):
                    # Validate the model on the entire evaluation test set after each epoch
                    output_loss =0
                    total_loss = 0
                    mse_loss=0
                    yhat = np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    yhateog =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    score = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    scoreeog = np.zeros([config.epoch_seq_len, len(gen.datalist), config.nclass])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt =np.zeros([config.epoch_seq_len, len(gen.datalist)])
                    feat_c = np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                    feat_eog= np.zeros([128,config.epoch_seq_len, len(gen.datalist)])

                    while test_step < num_batch_per_epoch-1:
                        (x_batch, y_batch) = gen[test_step]
                        x_c=x_batch[:,:,:,:,0:config.nchannel]
                        x_eog=x_batch[:,:,:,:,config.channel:config.channel+config.nchannel]
                        output_loss1, total_loss1, yhat1, score1, x_batch_eog = dev_step(x_eog, y_batch)
        
                        feat_eog[:, :, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(x_batch_eog)

                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhateog[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat1[n]
                            scoreeog[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size,:] = score1[n]

                        test_step += 1

                    if len(gen.datalist) > test_step*config.batch_size:
                        (x_batch, y_batch) = gen.get_rest_batch(test_step)
                        x_c=x_batch[:,:,:,:,0:config.nchannel]
                        x_eog=x_batch[:,:,:,:,config.channel:config.channel+config.nchannel]
                        output_loss1, total_loss1, yhat1, score1, x_batch_eog = dev_step(x_eog, y_batch)
        
                        feat_eog[:, :, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(x_batch_eog)
                        ygt[:, (test_step)*config.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhateog[n, (test_step)*config.batch_size : len(gen.datalist)] = yhat1[n]
                            scoreeog[n, (test_step)*config.batch_size : len(gen.datalist),:] = score1[n]

        
                    yhat = yhat + 1
                    ygt= ygt+1
                    yhateog+=1

                    acc1 = accuracy_score(ygt.flatten(), yhateog.flatten())
                    print(acc1)
                    return feat_eog, ygt, yhateog, scoreeog #feat_c, feat_eog, ygt, yhat, yhateog,score, scoreeog
        
                print('Test')
                train_feat, ygt1, yyhateog1,  scoreeog1 = evaluate(gen=test_generator)
                train_feat2, ygt, yyhateog2,  scoreeog2 = evaluate(gen=train_generator)

                savemat(os.path.join(config.out_dir2, "test_ret.mat"), dict(yhat = yyhateog1, acc = accuracy_score(ygt1.flatten(), yyhateog1.flatten()),kap=kap(ygt1.flatten(), yyhateog1.flatten()),
                                                                      ygt=ygt1, subjects=test_generator.subjects_datalist ,subjects2=test_generator.subjects, score=scoreeog1)  )              

                #load output probabilities (scores) on training dataset
                prob_train=np.load(config.out_dir2+ '/prob_train.npy')
                prob_train=prob_train[:,~np.isnan(np.nanmean(prob_train,0))]
                    
                    
                prob_train[np.where(prob_train==2)]=np.nan                
                
                #Make matrix to compare distances of test samples to train samples
                matr=np.zeros((test_generator.y.shape[0],train_generator.y.shape[0]))
                matr_umap=np.zeros((test_generator.y.shape[0],train_generator.y.shape[0]))
                train_feat2n=np.zeros((train_feat2.shape[0],train_generator.y.shape[0]))
                train_featn=np.zeros((train_feat.shape[0],test_generator.y.shape[0]))

                subjs=np.unique(train_generator.subjects_datalist)
                for s in subjs:
                    print(s)
                    tmp=train_feat2[:,:,np.array(train_generator.subjects_datalist)==s]  
                    diags = [np.mean(np.diagonal(np.flip(tmp,1), i,1,2),1) for i in range(-tmp.shape[1]+1,tmp.shape[2])]#-tmp.shape[1]+1)]
                    train_feat2n[:,np.array(train_generator.subjects)==s]=np.array(np.transpose(diags))
                subjs=np.unique(test_generator.subjects_datalist)
                for s in subjs:
                    print(s)
                    tmp=train_feat[:,:,np.array(test_generator.subjects_datalist)==s]  
                    diags = [np.mean(np.diagonal(np.flip(tmp,1), i,1,2),1) for i in range(-tmp.shape[1]+1,tmp.shape[2])]#-tmp.shape[1]+1)]
                    train_featn[:,np.array(test_generator.subjects)==s]=np.array(np.transpose(diags))
                train_feat2=train_feat2n#np.repeat(train_feat2n[:,np.newaxis,:],train_feat2.shape[1],axis=1)
                train_feat=train_featn#np.repeat(train_featn[:,np.newaxis,:],train_feat.shape[1],axis=1)


                print('Build matrix')
                for i in range(train_feat.shape[-1]): 
                    tmp1=np.sqrt(np.mean(np.power(train_feat[:,i:i+1]- train_feat2[:],2),0))
                    matr[i,:]=tmp1
                    
                print('Confidence, epistemic unc, aleatoric unc')
                conf=np.nanmean(prob_train,0)
                conf2=(conf-np.mean(conf,0))/np.std(conf,0) #normalize over dataset
                var= np.nanstd(prob_train,0) #epi unc 
                var2=(var-np.mean(var,0))/np.std(var,0)
                var_al= np.nanmean(prob_train*(1-prob_train),0)#ale unc
                var_al2=(var_al-np.mean(var_al,0))/np.std(var_al,0)
                

                #Most ambiguous data samples
                ambiguous1= np.argpartition(var,-int(math.floor(conf.shape[-1]*0.2)))[-int(math.floor(conf.shape[-1]*0.2)):]
                ambiguous2= np.argpartition(var_al,-int(math.floor(conf.shape[-1]*0.2)))[-int(math.floor(conf.shape[-1]*0.2)):]
                #Most easy data samples
                perc_labels=0.9
                easytoclass2= np.argpartition(-conf2+var_al2,int(math.floor(conf.shape[-1]*perc_labels)))[:int(math.floor(conf.shape[-1]*perc_labels))]

                #distance to closest class vs distance to second-closest class
                n=20
                mindistperclass2=np.zeros((train_feat.shape[-1],config.nclass))
                for i in np.arange(config.nclass)+1:
                    select=np.argmax(train_generator.y,1)==i-1
                    mindistperclass2[:,i-1]=np.mean(np.partition(matr[:,easytoclass2[select[easytoclass2]]], n)[:,:n],1)

                mindist2=np.mean(np.partition(matr[:,easytoclass2], n, axis=-1)[:,:n],1)
                mindistsamples2=np.argpartition(matr, n, axis=-1)[:,:n]
                mindistn = np.partition(matr, n, axis=-1)[:,:n]
                
                print('Compute uncertainty metrics')

                varepi_mindistsamples= var[mindistsamples2]#,1)
                varale_mindistsamples = var_al[mindistsamples2]#,1)
                conf_mindistsamples=conf[mindistsamples2]#,1)
                subjs=np.unique(train_generator.subjects_datalist)
                yyhateog22=np.zeros((len(train_generator.y), yyhateog2.shape[0]))
                ygt22=np.zeros((len(train_generator.y),yyhateog2.shape[0]))
                for s in subjs:
                    tmp=scoreeog2[:,np.array(train_generator.subjects_datalist)==s]#yyhateog1[:,dict3['subjects']==s]  
                    diags = [np.resize(np.diagonal(np.flip(tmp,0), i,0,1),(tmp.shape[-1],tmp.shape[0])) for i in range(-tmp.shape[0]+1,tmp.shape[1])]#-tmp.shape[1]+1)]
                    yyhateog22[np.array(train_generator.subjects)==s]=np.argmax(diags,1)
                    tmp=ygt[:,np.array(train_generator.subjects_datalist)==s]  
                    diags = [np.resize(np.diagonal(np.flip(tmp,0), i,0,1),tmp.shape[0]) for i in range(-tmp.shape[0]+1,tmp.shape[1])]#-tmp.shape[1]+1)]
                    ygt22[np.array(train_generator.subjects)==s]=np.array(diags)-1
                perf_mindistsamples2=np.mean(yyhateog22[mindistsamples2]==ygt22[mindistsamples2],(2))
                mindistperclass2=np.sort(mindistperclass2,1)
                distance_prop =mindistperclass2[:,0]/mindistperclass2[:,1]
                vals52=distance_prop
                sorted52= np.flipud(np.argsort(vals52))
                vals62=mindist2
                sorted62 = np.flipud(np.argsort(vals62)) #distance to closest training samples
                
                vals72= conf_mindistsamples
                sorted72= np.argsort(vals72)
                vals82=varale_mindistsamples
                sorted82= np.flipud(np.argsort(vals82))
                vals92=varepi_mindistsamples
                sorted92= np.flipud(np.argsort(vals92))
                vals102=perf_mindistsamples2
                sorted102= np.argsort(vals102)
                vals112 = mindistn
                
                subjs=np.unique(test_generator.subjects_datalist)
                vals00=np.zeros(( len(test_generator.subjects)))
                for s in subjs:
                    tmp=scoreeog1[:,np.array(test_generator.subjects_datalist)==s]  
                    diags = [np.mean(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])]
                    diags=np.array(diags)
                    _,vals0=cross_entropy(softmax(diags[:,:]),softmax(diags[:,:]))
                    vals00[np.array(test_generator.subjects)==s]=vals0
                sorted0=np.flipud(np.argsort(vals00))

                         
                thisdict = {
                  "entropy": sorted0,
                  "distance class proportion":sorted52,
                  "dist closest easy training": sorted62,
                  "conf closest training": sorted72,
                  "varale closest training": sorted82,
                  "varepi closest training": sorted92,
                  "perf closest training": sorted102
                 }
                np.save(config.out_dir2+'/sorted_DIAGS',thisdict)
                thisdict = {
                  "entropy": vals00,
                  "distance class proportion":vals52,
                  "dist closest easy training": vals62,
                  "conf closest training": vals72,
                  "varale closest training": vals82,
                  "varepi closest training": vals92,
                  "perf closest training": vals102,
                  "dist closest":vals112
                }
                np.save(config.out_dir2+'/vals_conf_DIAGS',thisdict)




###Figures paper

##Training dynamics
#V2 paper: used 46 patients osa (full EEG and fzfp2-wearable type data)
config.out_dir2=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accchans_subjnorm_sdtrain42pat/total/'
config.out_dir2='/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_fzfp2accchans2_subjnorm_sdtrain46pat/total0'
config.out_dir2='/esat/asterie1/scratch/ehereman/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_5chans2_subjnorm_sdtrain46pat/total0'
perc_labels=0.01
prob_train=np.load(config.out_dir2+ '/prob_train.npy')#[5:]
# score_test=np.load(config.out_dir2+ '/score_test.npy')
prob_train[np.where(prob_train==2)]=np.nan
# score_test[np.where(score_test==2)]=np.nan

conf=np.nanmean(prob_train,0)#[:,0]
conf2=(conf-np.mean(conf))/np.std(conf)
var= np.nanstd(prob_train,0)#[:,0]
var2=(var-np.mean(var))/np.std(var)
var_al= np.nanmean(prob_train*(1-prob_train),0)#[:,0]
var_al2=(var_al-np.mean(var_al))/np.std(var_al)
# varsim= np.sum(np.nanstd(np.max(softmax(score_test),-1),0),-1)#[:,0]                
# varsim_al= np.sum(np.nanmean(np.max(softmax(score_test),-1)*(1-np.max(softmax(score_test),-1)),0),-1)#[:,0]
# confsim=np.sum(np.nanmean(np.max(softmax(score_test),-1),0),-1)#[:,0]
ambiguous1= np.argpartition(var,-int(math.floor(conf.shape[-1]*perc_labels)))[-int(math.floor(conf.shape[-1]*perc_labels)):]
ambiguous2= np.argpartition(var_al,-int(math.floor(conf.shape[-1]*perc_labels)))[-int(math.floor(conf.shape[-1]*perc_labels)):]
hardtoclass1= np.argpartition(conf2+var2,int(math.floor(conf.shape[-1]*perc_labels)))[:int(math.floor(conf.shape[-1]*perc_labels))]
hardtoclass2= np.argpartition(conf2+var_al2,int(math.floor(conf.shape[-1]*perc_labels)))[:int(math.floor(conf.shape[-1]*perc_labels))]
easytoclass1= np.argpartition(-conf2+var2,int(math.floor(conf.shape[-1]*perc_labels)))[:int(math.floor(conf.shape[-1]*perc_labels))]
easytoclass2= np.argpartition(-conf2+var_al2,int(math.floor(conf.shape[-1]*perc_labels)))[:int(math.floor(conf.shape[-1]*perc_labels))]

ax = plt.subplot(111)
prob_train=np.load(config.out_dir2+ '/prob_train.npy')#[5:]

ax.plot(np.nanmean(prob_train,1),'k')
ax.plot(np.nanmean(prob_train[:,ambiguous1],1),'b')
ax.plot(np.nanmean(prob_train[:,ambiguous2],1),'g')
ax.plot(np.nanmean(prob_train[:,hardtoclass2],1),'r')
ax.plot(np.nanmean(prob_train[:,easytoclass2],1),'y')
ax.set_xlabel('Epoch')
ax.set_ylabel('Confidence')
ax.legend(['All','Model-ambiguous','Data-ambiguous','Hard-to-classify', 'Easy-to-classify'])

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(basepath+'/220427_confidenceAL/figures_paper/trainingdynamics_sleepuzl45pat.pdf')

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/trainingdynamics_sleepuzl45pat_5chans.pdf')


##UMAPS
import seaborn as sns
reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
trans= reducer.fit(train_feat2[:,5].reshape((128,-1)).transpose())
embedding2=trans.transform(train_feat2[:,5].reshape((128,-1)).transpose())
embedding =trans.transform(train_feat[:,5].reshape((128,-1)).transpose())

reducer= umap.UMAP(n_neighbors=30, min_dist=0.7)
trans= reducer.fit(train_feat2.reshape((128,-1)).transpose())
embedding2=trans.transform(train_feat2.reshape((128,-1)).transpose())
embedding =trans.transform(train_feat.reshape((128,-1)).transpose())

ygt2=np.argmax(train_generator.y,1)+1
colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt1[5].flatten()]
colors2=[sns.color_palette('hls',5)[int(i-1)] for i in ygt2[5].flatten()]

subjs=np.unique(test_generator.subjects_datalist)
yyhateog12=np.zeros((len(test_generator.y), yyhateog1.shape[0]))
ygt12=np.zeros((len(test_generator.y),yyhateog1.shape[0]))
for s in subjs:
    tmp=scoreeog1[:,np.array(test_generator.subjects_datalist)==s]#yyhateog1[:,dict3['subjects']==s]  
    diags = [np.resize(np.diagonal(np.flip(tmp,0), i,0,1),(tmp.shape[-1],tmp.shape[0])) for i in range(-tmp.shape[0]+1,tmp.shape[1])]#-tmp.shape[1]+1)]
    yyhateog12[np.array(test_generator.subjects)==s]=np.argmax(diags,1)
    tmp=ygt1[:,np.array(test_generator.subjects_datalist)==s]  
    diags = [np.resize(np.diagonal(np.flip(tmp,0), i,0,1),tmp.shape[0]) for i in range(-tmp.shape[0]+1,tmp.shape[1])]#-tmp.shape[1]+1)]
    ygt12[np.array(test_generator.subjects)==s]=np.array(diags)-1
colors=[sns.color_palette('hls',5)[int(i-1)] for i in ygt12[:,5]+1]
colors2=[sns.color_palette('hls',5)[int(i-1)] for i in ygt22[:,5]+1]


fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embedding[:,0], embedding[:,1],color=colors, s=3)
# ax.scatter(embedding[ambiguous1,0], embedding[ambiguous1,1],color=(0,0,0), s=20)
#ax.axis([-7,15,-9,13])
ax.axis('off')

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embedding2[:,0], embedding2[:,1],color=colors2, s=3)
# ax.scatter(embedding2[ambiguous1,0], embedding2[ambiguous1,1],color=(0,0,0), s=10)
#ax.axis([-7,15,-9,13])
ax.axis('off')


cc=var#(var_al-np.mean(var_al))/np.std(var_al) #var
import matplotlib as mpl
cmap = mpl.cm.Blues(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[4:,:-1])
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embedding2[:,0], embedding2[:,1],c=cc, s=3, cmap=cmap)#'Blues')#
ax.scatter(embedding2[ambiguous1,0], embedding2[ambiguous1,1],color=(0,0,0), s=6)
#ax.axis([-7,15,-9,13])
ax.axis('off')

fig, ax = plt.subplots()
# Create an image that will be used for the colorbar
cax = ax.imshow(cc.reshape(1,-1), aspect='auto', cmap=cmap)
# Add the colorbar to the plot
cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/colorbar_varepi.pdf')


cc=var_al#(var_al-np.mean(var_al))/np.std(var_al) #var
import matplotlib as mpl
cmap = mpl.cm.Greens(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[4:,:-1])
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(embedding2[:,0], embedding2[:,1],c=cc, s=3, cmap=cmap)#
ax.scatter(embedding2[ambiguous2,0], embedding2[ambiguous2,1],color=(0,0,0), s=6)
#ax.axis([-7,15,-9,13])
ax.axis('off')


fig, ax = plt.subplots()
# Create an image that will be used for the colorbar
cax = ax.imshow(cc.reshape(1,-1), aspect='auto', cmap=cmap)
# Add the colorbar to the plot
cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
plt.savefig(basepath+'/220427_confidenceAL/figures_paper/colorbar_varal.pdf')



#Find mean variance.
mean_var=[]
mean_varal=[]
mean_conf=[]
out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accchans2_subjnorm_sdtrain42pat'
# out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_fzfp2accchans2_subjnorm_sdtrain46pat'
for fold in range(23):
        path=out_dir+ '/n{:d}/group0/'.format(fold)
        path=out_dir+ '/total{:d}/'.format(fold)

        prob_train=np.load(path+ 'prob_train.npy')
        prob_train[np.where(prob_train==2)]=np.nan

        conf=np.mean(np.nanmean(prob_train,0),-1)#[:,0]
        var= np.mean(np.nanstd(prob_train,0),-1)#[:,0]
        var_al= np.mean(np.nanmean(prob_train*(1-prob_train),0),-1)#[:,0]

        mean_var.append(np.mean(var))
        mean_conf.append(np.mean(conf))
        mean_varal.append(np.mean(var_al))
print(np.mean(mean_var))
print(np.mean(mean_varal))
print(np.mean(mean_conf))



#Figure dataset cartography (bananas plot)
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

out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_1chans2_subjnorm_sdtrain42pat/total0/'
# out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2chans2_subjnorm_sdtrain42pat/total0/'
prob_train=np.load(out_dir+ 'prob_train.npy')
prob_train[np.where(prob_train==2)]=np.nan
conf=np.nanmean(prob_train,0)#np.mean(np.nanmean(prob_train,0),-1)#[:,0]
var= np.nanstd(prob_train,0)#np.mean(np.nanstd(prob_train,0),-1)#[:,0]
var_al= np.nanmean(prob_train*(1-prob_train),0)#np.mean(np.nanmean(prob_train*(1-prob_train),0),-1)#[:,0]
out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_3chans2_subjnorm_sdtrain42pat/total0/'
# out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accchans2_subjnorm_sdtrain42pat/total0/'
prob_train=np.load(out_dir+ 'prob_train.npy')
prob_train[np.where(prob_train==2)]=np.nan
conf2=np.nanmean(prob_train,0)#np.mean(np.nanmean(prob_train,0),-1)#[:,0]
var2= np.nanstd(prob_train,0)#np.mean(np.nanstd(prob_train,0),-1)#[:,0]
var_al2= np.nanmean(prob_train*(1-prob_train),0)#np.mean(np.nanmean(prob_train*(1-prob_train),0),-1)#[:,0]
out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_5chans2_subjnorm_sdtrain42pat/total0/'
# out_dir=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accgyrchans2_subjnorm_sdtrain42pat/total0/'
prob_train=np.load(out_dir+ 'prob_train.npy')
prob_train[np.where(prob_train==2)]=np.nan
conf3=np.nanmean(prob_train,0)#np.mean(np.nanmean(prob_train,0),-1)#[:,0]
var3= np.nanstd(prob_train,0)#np.mean(np.nanstd(prob_train,0),-1)#[:,0]
var_al3= np.nanmean(prob_train*(1-prob_train),0)#np.mean(np.nanmean(prob_train*(1-prob_train),0),-1)#[:,0]

x = var_al3
y =conf3

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
scat=ax.scatter(x, y, c=z, s=80, vmin=0,vmax=80)#80 voor psg
ax.set_xlabel('Aleatoric uncertainty')
ax.set_ylabel('Confidence')
# cbar=fig.colorbar(scat)


ax.set_xlim(left=-0.01, right=0.26,  emit=True, auto=False, xmin=None, xmax=None)
ax.set_ylim(-0.01, 1.01)#, emit=True, auto=False, xmin=None, xmax=None)
ax.set_xticks([0.1, 0.2])
ax.set_yticks([0,0.5, 1])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

values, xbins, ybins = np.histogram2d(x=var_al, y=conf,bins=10)
# values=values.T
xcenter=[]
ycenter=[]
arrowx=[]
arrowy=[]

for xbin in range(len(xbins)-1):
    for ybin in range(len(ybins)-1):
        if values[xbin,ybin]>50:
            tmo=(var_al>=xbins[xbin]) *( var_al<xbins[xbin+1])*(conf>=ybins[ybin])*(conf<ybins[ybin+1])
            assert(np.abs(np.sum(tmo)-values[xbin,ybin])<=1)
            xcenter.append((xbins[xbin]+xbins[xbin+1])/2)
            ycenter.append((ybins[ybin]+ybins[ybin+1])/2)
            
            arrowx.append(np.mean(var_al3[tmo]-var_al[tmo]))
            arrowy.append(np.mean(conf3[tmo]-conf[tmo]))
ax.quiver(xcenter, ycenter, arrowx, arrowy, scale=1)
 

           
fig, ax = plt.subplots()
scat=ax.scatter(x, y, c=z, s=100)
ax.set_xlabel('Aleatoric uncertainty')
ax.set_ylabel('Confidence')
cbar=fig.colorbar(scat)
r=np.arange(-0.8,0.8,0.01)
r0 = np.zeros(len(r))
ax.plot(r,r0,'k')
ax.plot(r0,r,'k')
ax.set_xlim(left=-0.21, right=0.21,  emit=True, auto=False, xmin=None, xmax=None)
ax.set_ylim(-0.81, 0.81)#, emit=True, auto=False, xmin=None, xmax=None)
# ax.set_xticks([0.1, 0.2])
# ax.set_yticks([0,0.5, 1])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

x = var
y =conf
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)
ax.set_xlabel('Epistemic uncertainty')
ax.set_ylabel('Confidence')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)