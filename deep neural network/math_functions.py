#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[4]:


def sigmoid(Z):
  
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache



# In[5]:


def relu(Z):
  
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


# In[6]:



def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[7]:


def relu_backward(dA, cache):

    
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ








