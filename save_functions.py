#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General functions to save data created

Created on Wed Dec  9 12:11:54 2020

@author: Elisabeth Heremans (ehereman)
"""
from datetime import datetime
import os


def save_neuralnetworkinfo(savepath, name, model, readme_text, originpath):
#    model.save(savepath+name) #--> only works with keras models
    readme = open(os.path.join(savepath , name + '.txt'), "w")
    now = datetime.now()
    now.strftime("%d/%m/%Y %H:%M:%S")
    readme.write('Model created on \n')
    readme.write(str(now))
    readme.write('\nfrom the code \n'+originpath )
    readme.write('\n \n')
    readme.write(readme_text)        
    readme.close()    
    
    
    
def save_fig_matplotlib(savepath, name, fig, readme_text,originpath):
    fig.savefig(savepath + name + '.png')
    readme = open(savepath + name + '.txt', "w")
    now = datetime.now()
    now.strftime("%d/%m/%Y %H:%M:%S")
    readme.write('Figure created on \n')
    readme.write(str(now))
    readme.write('from the code '+originpath )
    readme.write('\n \n')
    readme.write(readme_text)
    readme.close()

def print_instance_attributes(instance):
    '''
    From https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    '''
    stri=''
    for attribute, value in instance.__dict__.items():
        stri=stri+ attribute+ ' = '+ str(value)+ '\n'
    return stri