#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from math_functions import *
from DNN_functions import *
from load_data import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[29]:


train_x, train_y, test_x, test_y, classes = load_data()


# In[30]:


layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# In[32]:


parameters = L_layer_model(train_x, train_y, layers_dims,learning_rate = 0.009, num_iterations = 2500, print_cost = True)


# In[33]:


pred_train = predict(train_x, train_y, parameters)


# In[34]:


pred_test = predict(test_x, test_y, parameters)


# In[50]:


predict_image(parameters, classes, my_image = "cat.jpg")


# In[ ]:




