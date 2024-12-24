#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:57:21 2022

@author: ehereman
"""
import sys
import numpy as np
sys.path.insert(0, "/users/sista/ehereman/Documents/code/general")
from save_functions import *
from mathhelp import softmax


def uncertainty_metric(distances, confidences, alpha=1):
    dfd=softmax(distances)*confidences
    dfd=np.sum(dfd,1)
    fac=np.exp(alpha*np.mean(distances,1))
    return dfd*fac