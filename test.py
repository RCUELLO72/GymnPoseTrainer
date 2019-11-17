#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:16:47 2019

@author: rcuello
"""
from itertools import chain

def gl():
    r1 = range(6,16)
    r2 = range(70,91)
    r3 = range(121,128)
    return chain(r1,r2,r3)

for i in gl():
    print(i)
    
for i in gl():
    print(i)    