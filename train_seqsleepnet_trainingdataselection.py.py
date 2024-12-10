#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an adapted version of the code for training SeqSleepNet: 'https://github.com/pquochuy/SeqSleepNet/blob/master/tensorflow_net/SeqSleepNet/train_seqsleepnet.py'
Training SeqSleepNet with U-PASS:
saving the probabilities the network assigns 
to the training samples during training,
such that this can be used to infer which samples have high 
data uncertainty (aleatoric), or model uncertainty (epistemic).
+ 
Removing aleatoric (data) ambiguous training data
to see if this positively impacts downstream performance

Last adapted on Dec 9 2024.

@author: Elisabeth Heremans (ElisabethRMH)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf
import random
import scipy
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat

#from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math
basepath='...'
# sys.path.insert(1,basepath+'/Documents/code/adversarial_DA/')
# from adversarialnetwork import AdversarialNetwork
# from adversarialnet_config import Config

sys.path.insert(1,basepath+'/GitHub/SeqSleepNet/tensorflow_net/SeqSleepNet') #22/05: changed SeqSleepNet_E into SeqSleepNet
from seqsleepnet_sleep import SeqSleepNet_Sleep
from seqsleepnet_sleep_config import Config
sys.path.insert(1,basepath+'/Documents/code/feature_mapping/')
sys.path.insert(0, basepath+"/Documents/code/general")
from save_functions import *
from mathhelp import softmax, cross_entropy
sys.path.insert(0, basepath+"/Documents/code/adapted_tb_classes")
from subgenfromfile_epochsave import SubGenFromFile

filename=basepath+"/Documents/code/feature_mapping/sleepuzl_subsets_46pat.mat" #Fz or Fp2, they are in channel 5

files_folds=loadmat(filename)
normalize=True
source=basepath+'/SleepData/processed_tb/first69files-FzFp2sd-moreeegchan2/'
                
#root = '/esat/biomeddata/ehereman/MASS_toolbox'
number_patients=46
#VERSION WITH PATIENT GROUPS
for fold in range(23): 

    #for pat_group in range(1):#int(10/number_patients)):
                
        test_files=files_folds['test_sub'][0][fold][0]
        eval_files=files_folds['eval_sub'][0][fold][0]
        train_files=files_folds['train_sub'][0][fold][0]

        config= Config()
        config.epoch_seq_len=10
        config.epoch_step=config.epoch_seq_len
        
        
        test_generator=SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True,file_per_subject=True)

        train_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size,subjects_list=train_files,  sequence_size=config.epoch_seq_len,normalize_per_subject=True,file_per_subject=True)
        
        eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size,  subjects_list=eval_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True,file_per_subject=True)
       

        train_batches_per_epoch = np.floor(len(train_generator)).astype(np.uint32)
        eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
        test_batches_per_epoch = np.floor(len(test_generator)).astype(np.uint32)
        
        #E: nb of epochs in each set (in the sense of little half second windows)
        print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(len(train_generator.datalist), len(eval_generator.datalist), len(test_generator.datalist)))
        
        #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
        print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
        config.out_dir1= basepath+'/results_SeqSleepNet_tb/totalmass2/seqsleepnet_sleep_nfilter32_seq10_dropout0.75_nhidden64_att64_5chan_subjnorm_c4f4o2_160pat/total'#.format( fold+1)
        config.out_dir = basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl0001_fzfp2accchans2_subjnorm_sdtrain{:d}pat/total{:d}'.format(number_patients, fold)

        config.checkpoint_dir= './checkpoint/'
        config.allow_soft_placement=True
        config.log_device_placement=False
        config.nchannel=2
        config.domainclassifierstage2=False
        config.add_classifierinput=False
        config.out_path1= config.out_dir1
        config.out_path = config.out_dir
        config.domainclassifier = False  
        config.evaluate_every=200
        config.checkpoint_every = config.evaluate_every
        config.learning_rate= 1E-4
        config.training_epoch = 10# 20
        config.channel = 0
        config.perc_labels=0.001 #How much uncertain training data to remove
        config.TL=False
                    
        config.checkpoint_path1 = os.path.abspath(os.path.join(config.out_path1,config.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(config.out_path1)): os.makedirs(os.path.abspath(config.out_path1))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path1)): os.makedirs(os.path.abspath(config.checkpoint_path1))
        
        config.checkpoint_path = os.path.abspath(os.path.join(config.out_dir,config.checkpoint_dir))
        if not os.path.isdir(os.path.abspath(config.checkpoint_path)): os.makedirs(os.path.abspath(config.checkpoint_path))
            
        best_fscore = 0.0
        best_acc = 0.0
        best_kappa = 0.0
        min_loss = float("inf")
        # Training
        # ==================================================
        prob_tr=np.ones((config.training_epoch, len(train_generator.y)))*2#, config.epoch_seq_len))*2
        score_te=np.ones((int((config.training_epoch*train_batches_per_epoch/config.evaluate_every+1)), len(test_generator.y), config.nclass))*2#config.epoch_seq_len,
        
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
            session_conf = tf.compat.v1.ConfigProto(
              allow_soft_placement=config.allow_soft_placement,
              log_device_placement=config.log_device_placement,
              gpu_options=gpu_options)
            #session_conf.gpu_options.allow_growth = False
            sess = tf.compat.v1.Session(config=session_conf)
            with sess.as_default():
                net = SeqSleepNet_Sleep(config=config) #make seqsleepnet network
        
                # for batch normalization
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Define Training procedure
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    optimizer = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
                    grads_and_vars = optimizer.compute_gradients(net.loss)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
                out_dir = os.path.abspath(os.path.join(os.path.curdir,config.out_dir))
                print("Writing to {}\n".format(out_dir))
        
                saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
        
                if config.TL:
                    best_dir = os.path.join(config.checkpoint_path1, "best_model_acc")
                    saver.restore(sess, best_dir)
                    print("Model loaded")
                else:
                    sess.run(tf.compat.v1.global_variables_initializer())
        
                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                    epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                    feed_dict = {
                      net.input_x: x_batch,
                      net.input_y: y_batch,
                      net.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                      net.epoch_seq_len: epoch_seq_len,
                      net.frame_seq_len: frame_seq_len,
                      net.istraining: 1
                    }
                    _, step, output_loss, total_loss, accuracy,score = sess.run(
                       [train_op, global_step, net.output_loss, net.loss, net.accuracy, net.scores],
                       feed_dict)
                    return step, output_loss, total_loss, accuracy,score
        
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
                    output_loss, total_loss, yhat, scores = sess.run(
                           [net.output_loss, net.loss, net.predictions, net.scores], feed_dict)
                    return output_loss, total_loss, yhat, scores
        
                def evaluate(gen, log_filename):
                    # Validate the model on the entire evaluation test set after each epoch
                    datalstlen= len(gen.datalist)
                    output_loss =0
                    total_loss = 0
                    yhat = np.zeros([config.epoch_seq_len, datalstlen])
                    score= np.zeros([config.epoch_seq_len, datalstlen, config.nclass])
                    num_batch_per_epoch = len(gen)
                    test_step = 0
                    ygt = np.zeros([config.epoch_seq_len, datalstlen])
                    while test_step < num_batch_per_epoch-1:
                        (x_batch, y_batch) = gen[test_step]
                        x_batch=x_batch[:,:,:,:,config.channel:config.channel+config.nchannel]

                        output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                        output_loss += output_loss_
                        total_loss += total_loss_
                        ygt[:, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : (test_step+1)*config.batch_size] = yhat_[n]
                            score[n,(test_step)*config.batch_size : (test_step+1)*config.batch_size]=score_[n]
                        test_step += 1
                        
                    if datalstlen > (test_step)*config.batch_size:
                        (x_batch, y_batch) = gen.get_rest_batch(test_step)
                        x_batch=x_batch[:,:,:,:,config.channel:config.channel+config.nchannel]
                        print(x_batch.shape)
                        output_loss_, total_loss_, yhat_,score_ = dev_step(x_batch, y_batch)
                        ygt[:, (test_step)*config.batch_size : datalstlen] = np.transpose(np.argmax(y_batch,axis=2))
                        for n in range(config.epoch_seq_len):
                            yhat[n, (test_step)*config.batch_size : datalstlen] = yhat_[n]
                            score[n,(test_step)*config.batch_size : datalstlen]=score_[n]
                        output_loss += output_loss_
                        total_loss += total_loss_
                    ygt=ygt+1
                    yhat = yhat + 1
                    acc = 0
                    with open(os.path.join(out_dir, log_filename), "a") as text_file:
                        text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                        for n in range(config.epoch_seq_len):
                            acc_n = accuracy_score(yhat[n,:], ygt[n,:]) # due to zero-indexing
                            if n == config.epoch_seq_len - 1:
                                text_file.write("{:g} \n".format(acc_n))
                            else:
                                text_file.write("{:g} ".format(acc_n))
                            acc += acc_n
                    acc /= config.epoch_seq_len
                    return acc, yhat, ygt,output_loss, total_loss, score
        
                start_time = time.time()
                # Loop over number of epochs
                time_lst=[]
                outer_step=0
                epoch=0
                eval_acc, eval_yhat,eval_ygt, eval_output_loss, eval_total_loss,_ = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                test_acc, test_yhat, test_ygt,test_output_loss, test_total_loss,score = evaluate(gen=test_generator, log_filename="test_result_log.txt")
                # train_acc, train_yhat,train_ygt, train_output_loss, train_total_loss, train_score = evaluate(gen=train_generator, log_filename="train_result_log.txt")
                
                
                #removing high ambiguity/hard samples here.
                out_dir_=basepath+'/results_confidence/transferlearning/sleepuzl/46pat/seqslnet_cartotransferlearning_massc34tofzfp2_subjnorm_sdtrain42pat'
                out_dir_=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_cartotraining_fzfp2accchans2_subjnorm_sdtrain42pat'
                path_=out_dir_+ '/n{:d}/group0/'.format(fold,pat_group)
                path_=out_dir_+ '/total{:d}/'.format(fold,pat_group)
                # prob_test=np.load(path_+ 'prob_test.npy')
                prob_test=np.load(path_+ 'prob_train.npy')
                prob_test[np.where(prob_test==2)]=np.nan                

                conf=np.nanmean(prob_test,0)#np.sum(np.nanmean(prob_test,0),-1)#[:,0]
                conf2=(conf-np.mean(conf))/np.std(conf)
                var= np.nanstd(prob_test,0)#np.sum(np.nanstd(prob_test,0),-1)#[:,0]
                var2=(var-np.mean(var))/np.std(var)
                var_al= np.nanmean(prob_test*(1-prob_test),0)#np.sum(np.nanmean(prob_test*(1-prob_test),0),-1)#[:,0]
                var_al2=(var_al-np.mean(var_al))/np.std(var_al)
                ambiguous1= np.argpartition(var,-int(math.floor(conf.shape[-1]*config.perc_labels)))[-int(math.floor(conf.shape[-1]*config.perc_labels)):]
                ambiguous2= np.argpartition(var_al,-int(math.floor(conf.shape[-1]*config.perc_labels)))[-int(math.floor(conf.shape[-1]*config.perc_labels)):]
                hardtoclass1= np.argpartition(conf2+var2,int(math.floor(conf.shape[-1]*config.perc_labels)))[:int(math.floor(conf.shape[-1]*config.perc_labels))]
                hardtoclass2= np.argpartition(conf2+var_al2,int(math.floor(conf.shape[-1]*config.perc_labels)))[:int(math.floor(conf.shape[-1]*config.perc_labels))]
                easytoclass1= np.argpartition(-conf2+var2,int(math.floor(conf.shape[-1]*config.perc_labels)))[:int(math.floor(conf.shape[-1]*config.perc_labels))]
                easytoclass2= np.argpartition(-conf2+var_al2,int(math.floor(conf.shape[-1]*config.perc_labels)))[:int(math.floor(conf.shape[-1]*config.perc_labels))]
                
                low_conf_samples=ambiguous2

                #Simply putting the y to zero so that the backpropagated losses are zero for 
                # the uncertain training data that needs to be removed
                y_orig=train_generator.y.copy()
                train_generator.y[low_conf_samples]=0
                train_generator.shuffle=True
                train_generator.on_epoch_end()
                
                datalistind=test_generator.datalist#[step*config.batch_size:(step+1)*config.batch_size]                
                datalistind_all= [(np.array(datalistind)+i).tolist() for i in np.arange(config.epoch_seq_len)]                 
                datalistind_all2=[item for sublist in datalistind_all for item in sublist]
                scoress=np.moveaxis(np.array(score),0,1)
                scoress_all=np.reshape(scoress,(scoress.shape[0]*scoress.shape[1],-1),order='F')
                for i in np.unique(datalistind_all2):
                    inds=np.where(np.array(datalistind_all2)==i)[0]
                    scoress_all[inds]=np.mean(scoress_all[inds],0)
                
                score_te[0,datalistind_all2]=softmax(scoress_all)#np.transpose(score,axes=[1,0,2])
                
                train_batches_per_epoch=np.floor(len(train_generator)).astype(np.uint32)
                print('trainbatchesperepoch {:d}'.format( train_batches_per_epoch))

                while epoch < config.training_epoch:
                    epoch+=1
                    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                    step = 0
                    while step < train_batches_per_epoch:
                        # Get a batch
                        t1=time.time()
                        (x_batch, y_batch) = train_generator[step]
                        x_batch=x_batch[:,:,:,:,config.channel:config.channel+config.nchannel]
                        t2=time.time()
                        time_lst.append(t2-t1)
                        train_step_, train_output_loss_, train_total_loss_, train_acc_,scores_ = train_step(x_batch, y_batch)
                        time_str = datetime.now().isoformat()

                        #Storing the output probabilities while training
                        datalistind=train_generator.datalist[step*config.batch_size:(step+1)*config.batch_size]                        
                        datalistind_all= [(np.array(datalistind)+i).tolist() for i in np.arange(config.epoch_seq_len)]                 
                        datalistind_all2=[item for sublist in datalistind_all for item in sublist]
                        scoress=np.moveaxis(np.array(scores_),0,1)
                        scoress_all=np.reshape(scoress,(scoress.shape[0]*scoress.shape[1],-1),order='F')
                        for i in np.unique(datalistind_all2):
                            inds=np.where(np.array(datalistind_all2)==i)[0]
                            scoress_all[inds]=np.mean(scoress_all[inds],0)
                        y_original = y_orig[datalistind_all2]
                        prob_tr[epoch-1,datalistind_all2]=np.array([np.diagonal(softmax(scoress_all)[:,np.argmax(y_original,axis=1)])])[0]
    
        
                        # average acc
                        acc_ = 0
                        for n in range(config.epoch_seq_len):
                            acc_ += train_acc_[n]
                        acc_ /= config.epoch_seq_len
        
                        print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                        step += 1
        
                        current_step = tf.compat.v1.train.global_step(sess, global_step)
                        if current_step % config.evaluate_every == 0:
                            # Validate the model on the entire evaluation test set after each epoch
                            print("{} Start validation".format(datetime.now()))
                            eval_acc, eval_yhat,_, eval_output_loss, eval_total_loss,_ = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                            test_acc, test_yhat,_, test_output_loss, test_total_loss,_ = evaluate(gen=test_generator, log_filename="test_result_log.txt")


                            datalistind=test_generator.datalist#[step*config.batch_size:(step+1)*config.batch_size]                
                            datalistind_all= [(np.array(datalistind)+i).tolist() for i in np.arange(config.epoch_seq_len)]                 
                            datalistind_all2=[item for sublist in datalistind_all for item in sublist]
                            scoress=np.moveaxis(np.array(score),0,1)
                            scoress_all=np.reshape(scoress,(scoress.shape[0]*scoress.shape[1],-1),order='F')
                            for i in np.unique(datalistind_all2):
                                inds=np.where(np.array(datalistind_all2)==i)[0]
                                scoress_all[inds]=np.mean(scoress_all[inds],0)

                            score_te[((epoch-1)*train_batches_per_epoch+step)//config.evaluate_every, datalistind_all2]=softmax(scoress_all)#np.transpose(score, axes=[1,0,2])

        
                            if(eval_acc >= best_acc):
                                best_acc = eval_acc
                                checkpoint_name = os.path.join(config.checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                                save_path = saver.save(sess, checkpoint_name)
        
                                print("Best model updated")
                                print(checkpoint_name)
                                source_file = checkpoint_name
                                dest_file = os.path.join(config.checkpoint_path, 'best_model_acc')
                                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                                shutil.copy(source_file + '.index', dest_file + '.index')
                                shutil.copy(source_file + '.meta', dest_file + '.meta')
        
        

                    train_generator.on_epoch_end()
                    outer_step+=train_batches_per_epoch
                np.save(os.path.join(config.out_dir,'prob_train'),prob_tr)
                np.save(os.path.join(config.out_dir,'score_test'),score_te)
        
                end_time = time.time()
                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                    text_file.write("{:g}\n".format((end_time - start_time)))
                    text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
    
                save_neuralnetworkinfo(config.out_dir, 'TransfLearn',net,  originpath=__file__, readme_text=
                        'Transfer learning from seqslpnet network trained on MASS C4-A1 electrode montage to sleepuzl c4-a1  \n with subject normalization, LR of 1e-4, evaluate every 200 steps \n \n'.format(number_patients)+
                        'training length exp! 20 epochs.\n \n'.format(number_patients)+
                        'train on only random 10%  segments\n \n'.format(number_patients)+
                        print_instance_attributes(config))
                
