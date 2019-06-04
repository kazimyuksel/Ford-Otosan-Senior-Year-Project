# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:34:16 2019

@author: Kazım
"""
import h5py
with h5py.File('D://flipped_mean_safe.h5', 'w') as hf:
    hf.create_dataset("flipped_mean_safe",  data=flipped_mean_safe)
    
with h5py.File('D://flipped_mean_safe.h5', 'r') as hf:
    data = hf['flipped_mean_safe'][:]