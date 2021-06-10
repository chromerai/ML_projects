#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().system(' pip install opencv-python')


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 

get_ipython().run_line_magic('matplotlib', 'inline')
# to increase the size of images :
# possibility that it msy not come into efect at once so try to run it multiple (2-3-4) times
plt.rcParams['figure.figsize'] = (15,10)


# In[13]:


im = cv2.imread(r'data\asta.png')


# In[14]:


h,w,c = im.shape


# In[15]:


plt.imshow(im)
plt.show()


# In[16]:


image = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(np.hstack((im,image)))
plt.show()


# In[21]:


from sklearn.cluster import MiniBatchKMeans # difference between minibatch and normal kmean :- it takes a random sample out of all the data poinst it will figure out the mean points for all the whole sample . For eg
# it takes 500 points out of 10000 points randomly to determne the mean points , it is thus time and space effficient 


# In[17]:


flatimage = image.reshape(h*w,3)
flatimage.shape


# In[86]:


clt  = MiniBatchKMeans(n_clusters = 3)
labels = clt.fit_predict(flatimage)


# In[87]:


labels.shape


# In[88]:


clt.cluster_centers_


# SO we have seen that it return us floating point values **BUT** we know that pixels are always integers.So they need to be converted into integers , which is very easy to do and we know that each pixel can take values from 0-255 thus we can convert it into :
# 
# ### unsigned integer from range 0 - 255 . Thus we can do that by the following snipper of code :
# 
# clt.cluster_centers_.astype('uint8')
# 
# #### Also another important observation : 
# 
# Notice that the above and below matrixes are all centers of the 3 clusters we specified initially while defining clt . Each of these centroids(cause they are all 3 dimensional) contain 3 different values , corresponding to red , green , blue

# In[ ]:





# In[104]:


centroids = clt.cluster_centers_.astype('uint8')


# In[105]:


centroids[0]


# In[106]:


centroids[[2,1,2]]


# In[93]:


centroids[[2,2,2]]


# ***LABELS*** contain values from 0- k-1 (or here n_clusters-1)

# In[107]:


labels


# In[95]:


centroids.shape 


# In[96]:


result = centroids[labels]
result.shape


# In[97]:


result


# ### SO What does this result depicts ?????
# 
# Actually if u notice that when we equated 'result' with the above expression, actually the corresponding center for each of the 99457 points or data ,got mapped to their respective points and thus we see result shape as (999457,3) 

# In[108]:


resultImg  = result.reshape(h,w,c)
plt.imshow(resultImg)
plt.show()


# ## Notice the difference between the original and this new image :
# 
# Now , we see the development of some contours etc rather than the normal image as all the pixels are now dumped into only 5 clusters/colors unlike previously and thus , there is the development of contours , so as the value of k will increase the effect of contours will decrease and vice-versa

# We need to first convert our image to the format readable by normal image readers outside of opencv , thus we need to convert it back to its original order of color as we initially had  

# In[103]:


outImg = cv2.cvtColor(resultImg,cv2.COLOR_RGB2BGR)
cv2.imwrite('data/output_folder/compress.jpeg',outImg)


# In[ ]:





# In[ ]:




