'''
Active learning code.
Final query mechanism chosen was entr_epi. A few other possible uncertainty-based query mechanisms are shown in the code. 

'''

import copy
import umap
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1" 
import numpy as np
import tensorflow as tf
import math

import scipy
import random
import shutil, sys
from datetime import datetime
import h5py
import time
from scipy.io import loadmat, savemat
basepath='...'


sys.path.insert(0, basepath+"/Documents/code/adversarial_DA")
from adversarialnetwork_SeqSlNet_2nets_clean import AdversarialNet_SeqSlNet_2nets #code: see github project adversarial_DA

from config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kap


sys.path.insert(0, basepath+"/Documents/code/general")
from save_functions import *
from mathhelp import softmax, cross_entropy

sys.path.insert(0, basepath+"/Documents/code/adapted_tb_classes")
from subgenfromfile_epochsave import SubGenFromFile
from uncertainty_metrics import *

sys.path.insert(2,basepath+'/GitHub/Data-SUITE/')


filename=basepath+"/Documents/code/feature_mapping/sleepuzl_subsets_46pat.mat" #Fz or Fp2, they are in channel 5
files_folds_S=loadmat(filename)
filename=basepath+"/Documents/code/feature_mapping/sleepuzl_subsets_cross45pat.mat" #Fz or Fp2, they are in channel 5
files_folds=loadmat(filename)

normalize=True
source=basepath+'/SleepData/processed_tb/allfiles-FzFp2sd-moreeegchan2/'

def train_loop():

    number_patients=1
    #Patient per patient! Every fold = 3 patients, every pat_group =1 patient.
    for fold in range(15):
        for pat_group in range(3):
    
            test_files=[files_folds['test_sub'][0][fold][0][pat_group]]


            config= Config()
            config.epoch_seq_len=10
            config.epoch_step=config.epoch_seq_len
        
    
            eval_generator= SubGenFromFile(source,shuffle=False, batch_size=config.batch_size, subjects_list=test_files, sequence_size=config.epoch_seq_len, normalize_per_subject=True, file_per_subject=True)
            
           
            #The test recording here is also the recording on which we perform personalization. Hence, also used as training data. 
            # (most of it remains unlabeled of course, and with active learning we reveal a few labels of this data to the model)
            train_generator1= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size,subjects_list=test_files,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)

    
            # Parameters
            # ==================================================
            #E toevoeging
            FLAGS = tf.app.flags.FLAGS
            for attr, value in sorted(FLAGS.__flags.items()): # python3
                x= 'FLAGS.'+attr
                exec("del %s" % (x))
                
            
            # Misc Parameters
            tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
            
            tf.app.flags.DEFINE_string("out_dir1",out_dir1, "Point to output directory")
            tf.app.flags.DEFINE_string("out_dir", basepath+'/results_confidence/adv_DA/sleepuzl/15patcross/version4_45pat/seqslnet_advDA_AL10x1perc_'+sample_scheme+'4_sleepuzl5chans_subjnorm_sdtrain{:d}pat/total{:d}/n{:d}/group{:d}/'.format(number_patients,base_net_nb, fold,pat_group), "Point to output directory")
            tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
            
            tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
            
            tf.app.flags.DEFINE_integer("seq_len", 10, "Sequence length (default: 32)")
            
            tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
            
            tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
            tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 20)")
            
            tf.app.flags.DEFINE_integer('D',100,'Number of features') #new flag!
            
            FLAGS = tf.app.flags.FLAGS
            print("\nParameters:")
            for attr, value in sorted(FLAGS.__flags.items()): # python3
                print("{}={}".format(attr.upper(), value))
            print("")
            
            # Data Preparatopn
            # ==================================================
            
            # path where some output are stored
            out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
            # path where checkpoint models are stored
            checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
            if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
            if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))
    
            out_path1= FLAGS.out_dir1 
            checkpoint_path1 = os.path.abspath(os.path.join(out_path1,FLAGS.checkpoint_dir))
            if not os.path.isdir(os.path.abspath(out_path1)): os.makedirs(os.path.abspath(out_path1))
            if not os.path.isdir(os.path.abspath(checkpoint_path1)): os.makedirs(os.path.abspath(checkpoint_path1))
            config.out_path1=out_path1
            config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
            config.epoch_seq_len = FLAGS.seq_len
            config.epoch_step = FLAGS.seq_len
            config.nfilter = FLAGS.nfilter
            config.nhidden1 = FLAGS.nhidden1
            config.attention_size1 = FLAGS.attention_size1
            config.nchannel = 2
            config.training_epoch = int(10) 
            config.same_network=True
            config.feature_extractor=True
            config.learning_rate=1e-4
            config.mult_channel=False
            config.withEOGclass=True#!!!!!
            config.channel =0
            config.channels_dataaugm=[]
            config.add_classifieroutput=False
            config.GANloss=True
            config.domain_lambda= 0.01
            config.fix_sourceclassifier=False
            config.domainclassifier=True
            config.shareDC=False
            config.shareLC=False
            config.mmd_weight=1
            config.mmd_loss=False
            config.DCweighting=False
            config.SNweighting=False
            config.pseudolabels = False
            config.DCweightingpslab=False
            config.SNweightingpslab=False
            config.weightpslab=0.01
            config.crossentropy=False
            config.classheads2=False
            config.advdropout=False
            config.evaluate_every=100
            config.adversarialentropymin=False
            config.minneighbordiff=False
            config.subjectclassifier = False   
            config.diffattn=False
            config.diffepochrnn=False
            config.regtargetnet=False
            config.KLregularization=False
            config.kl_reg_lambda=0.4
            config.T=1
            config.perc_labels= 0.1#0.01#0.01
            config.targetclassweight=2.5#2.5
            config.epoch_add_data=1#10
            
            train_batches_per_epoch = np.floor(len(train_generator1)).astype(np.uint32)
            eval_batches_per_epoch = np.floor(len(eval_generator)).astype(np.uint32)
            

            print("Train/Eval/Test set: {:d}/{:d}".format(len(train_generator1.datalist), len(eval_generator.datalist)))
            
            #E: nb of batches to run through whole dataset = nb of sequences (20 consecutive epochs) divided by batch size
            print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))
            
            
            
            # variable to keep track of best fscore
            best_fscore = 0.0
            best_acc = 0.0
            best_train_total_loss=np.inf
            best_kappa = 0.0
            min_loss = float("inf")
            # Training
            # ==================================================
            
            with tf.Graph().as_default():
                session_conf = tf.ConfigProto(
                  allow_soft_placement=FLAGS.allow_soft_placement,
                  log_device_placement=FLAGS.log_device_placement)
                session_conf.gpu_options.allow_growth = True
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    arnn=AdversarialNet_SeqSlNet_2nets(config)
            
                    # Define Training procedure
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    optimizer = tf.train.AdamOptimizer(config.learning_rate)
                    
                    
                    domainclass_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'domainclassifier_net')

                    excl=[]
                    allvars= [var for var in tf.trainable_variables() if (var not in domainclass_vars and var not in excl)]
                    grads_and_vars = optimizer.compute_gradients(arnn.loss, var_list=allvars)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    if config.GANloss:
                        global_step2 = tf.Variable(0, name="global_step2", trainable=False)
                        grads_and_vars2 = optimizer.compute_gradients(arnn.domainclass_loss_sum, var_list = domainclass_vars)
                        train_op2 = optimizer.apply_gradients(grads_and_vars2, global_step=global_step2)
    
            
            
                    out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
                    print("Writing to {}\n".format(out_dir))
            
                    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer/output-'),max_to_keep=1)

                    # initialize all variables
                    sess.run(tf.initialize_all_variables())
                    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer/output-'),max_to_keep=1)

                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    var_list1 = {}
                    for v1 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'output_layer/output'):
                        tmp = v1.name.replace(v1.name[0:v1.name.index('-')],'output_layer/output')
                        tmp=tmp[:-2]
                        var_list1[tmp]=v1
                    saver1=tf.train.Saver(var_list=var_list1)                    
                    saver1.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    
                    var_list2= {}
                    for v2 in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seqsleepnet_c34'):
                        
                        tmp=v2.name[16:-2]
                        if tmp[0]=='2':
                            continue
                        var_list2[tmp]=v2
                    saver2=tf.train.Saver(var_list=var_list2)
                    saver2.restore(sess, os.path.join(checkpoint_path1, "best_model_acc"))
                    

                    saver1 = tf.train.Saver(tf.all_variables(), max_to_keep=1)
                    print("Model loaded")
            
                    def train_step(x_batch, y_batch,trg_bool, src_bool): #not adapted
                        """
                        A single training step
                        """
                        frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                        epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                        feed_dict = {
                          arnn.trg_bool:trg_bool,
                          arnn.src_bool: src_bool,
                          arnn.input_x: x_batch,
                          arnn.input_y: y_batch,
                          arnn.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                          arnn.frame_seq_len: frame_seq_len,
                          arnn.epoch_seq_len: epoch_seq_len,
                          arnn.training: True,
                          arnn.weightpslab:config.weightpslab
                        }
                        if config.GANloss:

                            _,_, step, output_loss, output_loss_eog, domain_loss,total_loss, accuracy,scores = sess.run(
                                [train_op, train_op2, global_step, arnn.output_loss, arnn.output_loss_eog, arnn.domain_loss_sum, arnn.loss, arnn.accuracy, arnn.scores_eog],
                                feed_dict)
                                
                        else:

                            _, step, output_loss,output_loss_eog, domain_loss, total_loss, accuracy, scores = sess.run(
                                [train_op, global_step, arnn.output_loss,arnn.output_loss_eog, arnn.domain_loss_sum, arnn.loss, arnn.accuracy, arnn.scores_eog],
                                feed_dict)
                        return step, output_loss,output_loss_eog, domain_loss, total_loss, np.mean(accuracy), scores
            
                    def dev_step(x_batch, y_batch, dropout=False):
                        frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len,dtype=int) * config.frame_seq_len
                        epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
                        dropout_keep=1.0
                        if dropout is not False:
                            dropout_keep=dropout
                        feed_dict = {
                            arnn.trg_bool:np.ones(len(x_batch)),
                            arnn.src_bool: np.ones(len(x_batch)),
                            arnn.input_x: x_batch,
                            arnn.input_y: y_batch,
                            arnn.dropout_keep_prob_rnn: dropout_keep,
                            arnn.frame_seq_len: frame_seq_len,
                            arnn.epoch_seq_len: epoch_seq_len,
                            arnn.training: dropout is not False,
                            arnn.weightpslab:config.weightpslab
                        }

                        output_loss, output_loss_eog, domain_loss, total_loss, yhat, yhateog, yhatD , score, features= sess.run(
                                [arnn.output_loss, arnn.output_loss_eog, arnn.domain_loss_sum, arnn.loss, arnn.predictions, arnn.predictions_eog, arnn.predictions_D, arnn.scores_eog, arnn.features2], feed_dict)
                        return output_loss, output_loss_eog, domain_loss, total_loss, yhat, yhateog, yhatD, score, features
            
            
                    def evaluate(gen, log_filename,dropout=False,low_conf_samples=[]):
                        # Validate the model on the entire evaluation test set after each epoch
                        datalstlen=len(gen.datalist)
                        output_loss =0
                        total_loss = 0
                        domain_loss=0
                        output_loss_eog=0
                        yhat = np.zeros([config.epoch_seq_len, datalstlen])
                        yhateog = np.zeros([config.epoch_seq_len, datalstlen])
                        num_batch_per_epoch = len(gen)
                        test_step = 0
                        ygt = np.zeros([config.epoch_seq_len, datalstlen])
                        feat = np.zeros([128,config.epoch_seq_len, len(gen.datalist)])
                        yhatD = np.zeros([config.epoch_seq_len, datalstlen*2])
                        yd= np.zeros([config.epoch_seq_len, datalstlen*2])
                        score= np.zeros([config.epoch_seq_len, datalstlen, config.nclass])
                        while test_step < num_batch_per_epoch-1:    
    
                            (x_batch,y_batch)=gen[test_step]
                            x_batch=x_batch[:,:,:,:,np.concatenate((np.arange(config.channel,config.channel+config.nchannel), np.arange(config.channel,config.channel+config.nchannel)))]
    
            
                            output_loss_,output_loss_eog_, domain_loss_, total_loss_, yhat_, yhateog_, yhatD_,score_, features1_ = dev_step(x_batch, y_batch, dropout)
                            ygt[:, (test_step)*gen.batch_size : (test_step+1)*gen.batch_size] = np.transpose(np.argmax(y_batch,axis=2))
                            yd[:,(test_step)*gen.batch_size*2 : (test_step+1)*gen.batch_size*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                            feat[:,:, (test_step)*gen.batch_size : (test_step+1)*gen.batch_size] = np.transpose(features1_)
                            for n in range(config.epoch_seq_len):
                                yhat[n, (test_step)*gen.batch_size : (test_step+1)*gen.batch_size] = yhat_[n]
                                yhateog[n, (test_step)*gen.batch_size : (test_step+1)*gen.batch_size] = yhateog_[n]
                                yhatD[n, (test_step)*gen.batch_size*2 : (test_step+1)*gen.batch_size*2] = yhatD_[n]
                                score[n,(test_step)*gen.batch_size : (test_step+1)*gen.batch_size]=score_[n]
                            output_loss += output_loss_
                            total_loss += total_loss_
                            domain_loss += domain_loss_
                            output_loss_eog += output_loss_eog_
                            test_step += 1

                                
                        if len(gen.datalist) > test_step*gen.batch_size:
                            # if using load_random_tuple
                            (x_batch,y_batch)=gen.get_rest_batch(test_step)
                            x_batch=x_batch[:,:,:,:,np.concatenate((np.arange(config.channel,config.channel+config.nchannel), np.arange(config.channel,config.channel+config.nchannel)))]     
            
                            output_loss_, output_loss_eog_, domain_loss_, total_loss_, yhat_, yhateog_, yhatD_,score_ , features1_= dev_step(x_batch, y_batch,dropout)
                            ygt[:, (test_step)*gen.batch_size : len(gen.datalist)] = np.transpose(np.argmax(y_batch,axis=2))
                            yd[:,(test_step)*gen.batch_size*2 : len(gen.datalist)*2] = np.concatenate([np.ones(len(x_batch)), np.zeros(len(x_batch))])
                            feat[:,:, (test_step)*gen.batch_size : len(gen.datalist)] = np.transpose(features1_)
                            for n in range(config.epoch_seq_len):
                                yhat[n, (test_step)*gen.batch_size : len(gen.datalist)] = yhat_[n]
                                yhateog[n, (test_step)*gen.batch_size : len(gen.datalist)] = yhateog_[n]
                                yhatD[n, (test_step)*gen.batch_size*2 : len(gen.datalist)*2] = yhatD_[n]
                                score[n,(test_step)*gen.batch_size : datalstlen]=score_[n]
                            output_loss += output_loss_
                            domain_loss += domain_loss_
                            total_loss += total_loss_
                            output_loss_eog += output_loss_eog_
                        yhat = yhat + 1
                        ygt= ygt+1
                        yhateog+=1
                        acc = accuracy_score(ygt.flatten(), yhat.flatten())
                        
                        if not gen.shuffle:
                            diags00=np.zeros(gen.y.shape[0])
                            diags01=np.zeros(gen.y.shape[0])
                            tmp=score
                            diags =np.array( [np.sum(np.diagonal(np.flip(tmp,0),i),1) for i in range(-tmp.shape[0]+1,tmp.shape[1])])
                            diags00=np.argmax(diags,1)
                            tmp=ygt#[:,dict3['subjects']==s]  
                            diags = np.array([np.mean(np.mean(np.diagonal(np.flip(tmp,0),i))) for i in range(-tmp.shape[0]+1,tmp.shape[1])])
                            diags01=np.array(diags)-1
                            accEOG= accuracy_score(diags00.flatten(), diags01.flatten())

                            labeled_bool =np.ones(gen.y.shape[0])
                            labeled_bool[low_conf_samples]=0
                            labeled_bool=labeled_bool.astype(bool)                                    
    
                            acc_unl=  accuracy_score(diags00[labeled_bool].flatten(), diags01[labeled_bool].flatten())
                        else:
                            accEOG= accuracy_score(ygt.flatten(), yhateog.flatten())
                            acc_unl=0.0
                        
                        if len(log_filename)>0:
                            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                                text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} {:g} \n".format(output_loss, output_loss_eog, domain_loss, total_loss, acc, accEOG, acc_unl))
                        return accEOG, yhateog, ygt, output_loss, output_loss_eog, total_loss, score, feat

                    def calc_entropydiags(generator):
                        '''Calculate entropy for every sample. 
                        Bc of the sequence-to-sequence approahc of SeqSleepnet, 
                        each sample has seq_len evaluations, so we take an average over these'''

                        train_acc, train_yhat, train_ygt, train_output_loss, train_output_loss_eog, train_total_loss, train_score,train_feat = evaluate(gen=generator, log_filename="")
                        _,a=cross_entropy(softmax(train_score/config.T),softmax( train_score/config.T))

                        vals=np.zeros(generator.y.shape[0])
                        subjs=np.unique(generator.subjects)
                        for s in subjs:
                            sub_idx=np.where(np.array(generator.subjects_datalist)==s)[0]
                            diags = np.array([np.mean(a[::-1,sub_idx].diagonal(i)) for i in range(-a.shape[0]+1,len(sub_idx))])#[all_samples]
                            sub_idx=np.where(np.array(generator.subjects)==s)[0]                            
                            vals[sub_idx]=diags
                        return vals
                    
                    start_time = time.time()
                    time_lst=[]
            
                    # Loop over number of epochs
                    eval_acc, eval_yhat, eval_ygt, eval_output_loss,eval_output_loss_eog, eval_total_loss,eval_score ,_= evaluate(gen=eval_generator, log_filename="eval_result_log.txt")

                    train_generator1.y=np.zeros(train_generator1.y.shape)
                    low_conf_samples=[]
                    train_batches_per_epoch=np.floor(len(train_generator1)).astype(np.uint32)
                    print('trainbatchesperepoch {:d}'.format( train_batches_per_epoch))
                    prob_tr=np.ones((config.training_epoch, len(eval_generator.y)))*np.nan#, config.epoch_seq_len))*2
                    score_tr=np.ones((config.training_epoch, len(eval_generator.y),config.nclass))*np.nan
                    score_tr_unlab=np.ones((config.training_epoch, len(eval_generator.y),config.nclass))*np.nan
                    
                    for epoch in range(config.training_epoch):
                        new_low_conf_samples=[]
                        if epoch<config.epoch_add_data:
                            all_samples=list(np.arange(eval_generator.y.shape[0]))
                            for l in np.array(low_conf_samples):
                                all_samples.remove(l)
                            
                            # A NUMBER OF APPROACHES FOR SAMPLING WERE TESTED. Entr_epi was the approach used in the final paper

                            if sample_scheme=='rndm':
                                new_low_conf_samples=random.sample(all_samples,int(math.floor(len(eval_generator.y)*config.perc_labels)))
                            elif sample_scheme=='entr_diags' or ('entr' in sample_scheme and epoch==0):
                                vals=calc_entropydiags(eval_generator)
                                vals=vals[all_samples]
                                selected=np.argpartition(vals,-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)))[-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)):]
                                new_low_conf_samples=np.array(all_samples)[selected]
                            elif sample_scheme=='prob_diags':
                                train_acc, train_yhat, train_ygt, train_output_loss, train_output_loss_eog, train_total_loss, train_score,train_feat = evaluate(gen=eval_generator, log_filename="")
                                dfd=np.max(softmax(train_score),2)

                                vals=np.zeros(train_generator1.y.shape[0])
                                subjs=np.unique(eval_generator.subjects)
                                for s in subjs:
                                    sub_idx=np.where(np.array(eval_generator.subjects_datalist)==s)[0]
                                    diags = np.array([np.mean(dfd[::-1,sub_idx].diagonal(i)) for i in range(-dfd.shape[0]+1,len(sub_idx))])#[all_samples]
                                    sub_idx=np.where(np.array(eval_generator.subjects)==s)[0]                            
                                    vals[sub_idx]=diags
                                
                                vals=-vals[all_samples]
                                selected=np.argpartition(vals,-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)))[-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)):]
                                new_low_conf_samples=np.array(all_samples)[selected]
                            
                            elif 'entr' in sample_scheme:
                                
                                train_acc, train_yhat, train_ygt, train_output_loss, train_output_loss_eog, train_total_loss, train_score,train_feat = evaluate(gen=eval_generator, log_filename="")
                                _,a=cross_entropy(score_tr_unlab[:epoch]/config.T, score_tr_unlab[:epoch]/config.T)
                                _,vals=cross_entropy(np.mean(score_tr_unlab[:epoch],0), np.mean(score_tr_unlab[:epoch],0))

                                if 'tot' in sample_scheme or epoch<=1:
                                    vals=vals
                                elif 'ale' in sample_scheme:
                                    vals=np.nanmean(a,0)
                                elif 'epi' in sample_scheme: #THIS IS THE APPROACH WE TOOK IN FINAL PAPER
                                    vals=vals-np.nanmean(a,0)                                       
                                    
                                vals=vals[all_samples]
                                selected=np.argpartition(vals,-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)))[-int(math.floor(eval_generator.y.shape[0]*config.perc_labels)):]
                                new_low_conf_samples=np.array(all_samples)[selected]
                            

                        low_conf_samples.extend( list(new_low_conf_samples))
                        
                        y2=train_generator1.y
                        y2[low_conf_samples]=eval_generator.y[low_conf_samples]
                        train_generator1.y=y2
    
                        if epoch>=10 and config.weightpslab==0:
                            config.weightpslab==0.1
                        
                        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                        step = 0
                        while step < train_batches_per_epoch:
                            # Get a batch
                            t1=time.time()        

                            #batch 3: Batch of other recordings (train_generator, source dataset)
                            (x_batch3,y_batch3)=train_generator[step]
                            x_batch3= x_batch3[:,:,:,:,np.arange(config.channel, config.channel+config.nchannel)]
                            x_batch3=np.append(x_batch3,np.zeros(x_batch3.shape),axis=-1)
                            
                            # #batch 1: Batch of relevant recording (train_generator1, target dataset)
                            (x_batch1,y_batch1)=train_generator1[step]
                            chnl=np.arange(config.channel, config.channel+config.nchannel)
                            x_batch1=x_batch1[:,:,:,:,chnl]                     
                            x_batch1=np.append(x_batch1,x_batch1,axis=-1) ##adapt 3: np.zeros(x_batch1.shape)
                            

                            x_batch0=np.vstack([x_batch1, x_batch3]) #X_batch
                            y_batch0=np.vstack([y_batch1, y_batch3])  #y_batch
                            
                            
                            t2=time.time()
                            time_lst.append(t2-t1)                        
                            src_bool=np.concatenate([ np.ones(len(x_batch1)),  np.zeros(len(x_batch3))])
                            trg_bool = np.concatenate([ np.zeros(len(x_batch1)), np.ones(len(x_batch3))])
                            
                            train_step_, train_output_loss_, train_output_loss_eog_,train_domain_loss_, train_total_loss_, train_acc_, scores_ = train_step(x_batch0, y_batch0, trg_bool, src_bool)
                            
                            time_str = datetime.now().isoformat()
            
                            print("{}: step {}, output_loss {}, output_loss_eog {}, domain_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_output_loss_eog_, train_domain_loss_, train_total_loss_, train_acc_))
                            step += 1
            
                            current_step = tf.train.global_step(sess, global_step)
                            
            
                        train_generator.on_epoch_end()
                        train_generator1.on_epoch_end()
                        
                        eval_acc, yyhateog1, ygt1, eval_output_loss,eval_output_loss_eog, eval_total_loss,scoreeog1,_ = evaluate(gen=eval_generator, log_filename="eval_result_log.txt",low_conf_samples=low_conf_samples)

                        tmp=scoreeog1 
                        diags = np.array([np.mean(np.diagonal(np.flip(tmp,0), i,0,1),-1) for i in range(-tmp.shape[0]+1,tmp.shape[1])])#-tmp.shape[1]+1)]

                        y_labeled=eval_generator.y[low_conf_samples]
                        prob_tr[epoch, low_conf_samples]=np.diagonal(softmax(diags[low_conf_samples])[:,np.argmax(y_labeled,axis=1)])
                        score_tr[epoch, low_conf_samples]=softmax(diags[low_conf_samples])
                        score_tr_unlab[epoch]=softmax(diags)

                    
                    # best_acc = eval_acc
                    checkpoint_name = os.path.join(checkpoint_path, 'best_model_acc')
                    save_path = saver.save(sess, checkpoint_name)
    

                    savemat(os.path.join(out_path, "test_retFzFp2.mat"), dict(yhat = yyhateog1, acc = accuracy_score(ygt1.flatten(), yyhateog1.flatten()),kap=kap(ygt1.flatten(), yyhateog1.flatten()),
                                                                          ygt=ygt1, subjects=eval_generator.subjects_datalist, score=scoreeog1)  )              


                    np.save(os.path.join(out_dir,'prob_train'),prob_tr)
                    np.save(os.path.join(out_dir,'score_train'),score_tr)
                    np.save(os.path.join(out_dir,'score_train_unlab'),score_tr_unlab)
                    
                    np.save(out_dir+'/labeledsamples',low_conf_samples)
                    end_time = time.time()
                    with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                        text_file.write("{:g}\n".format((end_time - start_time)))
                        text_file.write("mean generator loading time {:g}\n".format((np.mean(time_lst))))
        
                    save_neuralnetworkinfo(checkpoint_path, 'fmandclassnetwork',arnn,  originpath=__file__, readme_text=
                            'Domain adaptation unsup personalization with GAN loss and classification network on sleep uzl (with normalization per patient) \n c34 net and eog net are same, initialized with SeqSleepNet trained on mass\n'+
                            'training on {:d} patients \n validation with pseudo-label accuracy \n no batch norm \n baseline net is trained SleepUZL training set \n batch size 32, WITH eog classifier, save net at end and early stop at best test pseudolabel acc on EOG. LR 1e-4,  \n'.format(number_patients)+
                            ' with eog classification layer. not fixed source classifier. fixed batch size 32 for source and target!!!  \n'+# \n with pseudolabels: for unmatched EOG, we use labels from source net classifier for training target net classifier. NOT weighted. \n'+
                            '\n unlabeled unmatched fzfp2 data from sleep uzl \n using fzfp2 of sleepuzl train patients in source, only fp2fz of sleepuzl 1 pat in target classifier \n no overlap between patients source/target. \n\n'+
                            'in this version, eval every 200 steps & only 10 epochs,  \n '+
                            'train only with first 20 labels (unshuffled) \n \n'+
                            print_instance_attributes(config))


for base_net_nb in range(1):
    print('basenetnb '+str(base_net_nb))
    config= Config()
    config.epoch_seq_len=10
    config.epoch_step=config.epoch_seq_len
    config.batch_size=32
    config.var_thresh=0.8#=0.9
    
    train_files1=files_folds_S['train_sub'][0][base_net_nb][0]
    train_generator= SubGenFromFile(source,shuffle=True, batch_size=config.batch_size,subjects_list=train_files1,  sequence_size=config.epoch_seq_len,normalize_per_subject=True, file_per_subject=True)

    out_dir1=basepath+'/results_confidence/scratch/sleepuzl/46pat/seqslnet_learning_excambigAl001_5chans2_subjnorm_sdtrain46pat/total{:d}'.format(base_net_nb)
    for sample_scheme in ['entr_epi']:#['entr_epi','entr_diags']:
                meanvar_app=[]
                meanconf_app=[]
                
                meanentr_app=[]
                meanprob_app=[]
                train_loop()
