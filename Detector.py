#!/usr/bin/env python
# coding: utf-8

# In[106]:


import os

base_path = 'dataset'
images = os.path.sep.join([base_path,'images'])
annotations = os.path.sep.join([base_path,'airplanes.csv'])


# In[107]:


# load dataset

rows = open(annotations).read().strip().split("\n")

data = []
targets = []
filenames = []


# In[108]:


# spliting dataset
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
for row in rows:
    row = row.split(",")
    (filename,startX,startY,endX,endY) = row
    
    imagepaths = os.path.sep.join([images,filename])
    image = cv2.imread(imagepaths)
    (h,w) = image.shape[:2]
    
    startX = float(startX) / w
    startY = float(startY) / h
    
    endX = float(endX) / w
    endY = float(endY) / h
    
    image = load_img(imagepaths,target_size=(224,224))
    image = img_to_array(image)
    
    targets.append((startX,startY,endX,endY))
    filenames.append(filename)
    data.append(image)


# In[109]:


# Normalizing the Dataset
import numpy as np
data = np.array(data,dtype='float32') / 255.0
targets = np.array(targets,dtype='float32')


# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


# splitting into training and testing
split = train_test_split(data,targets,filenames,test_size=0.10,random_state = 42)


# In[112]:


(train_images,test_images) = split[:2]
(train_targets,test_targets) = split[2:4]
(train_filenames,test_filenames) = split[4:]


# In[113]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input


# In[114]:


vgg = VGG16(weights='imagenet',include_top = False,input_tensor=Input(shape=(224,224,3)))


# In[115]:


vgg.summary()


# In[116]:


from tensorflow.keras.layers import Input,Flatten,Dense


# In[117]:


vgg.trainable = False

flatten = vgg.output

flatten = Flatten()(flatten)


# In[118]:


bboxhead = Dense(128,activation="relu")(flatten)
bboxhead = Dense(64,activation="relu")(bboxhead)
bboxhead = Dense(32,activation="relu")(bboxhead)
bboxhead = Dense(4,activation="relu")(bboxhead)


# In[119]:


from tensorflow.keras.models import Model


# In[120]:


model = Model(inputs = vgg.input,outputs = bboxhead)


# In[121]:


model.summary()


# In[122]:


from tensorflow.keras.optimizers import Adam

opt = Adam(1e-4)


# In[123]:


model.compile(loss='mse',optimizer=opt)


# In[124]:


# history = model.fit(train_images,train_targets,validation_data=(test_images,test_targets),batch_size=32,epochs=10,verbose=1)


# In[125]:


from tensorflow.keras.models import load_model


# In[126]:


model = load_model('detector.h5')


# In[127]:


imagepath = 'dataset/images/image_0133.jpg'


# In[128]:


image = load_img(imagepath,
                 target_size=(224,224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image,axis=0)


# In[129]:


preds = model.predict(image)[0]
(startX,startY,endX,endY) = preds


# In[130]:


import imutils


# In[131]:


image = cv2.imread(imagepath)
image = imutils.resize(image,width=600)


# In[132]:


(h,w) = image.shape[:2]


# In[133]:


startX = int(startX * w)
startY = int(startY * h)

endX = int(endX *w)
endY = int(endY * h)


# In[134]:


cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),3)


# In[135]:


cv2.imshow('output',image)
cv2.waitKey(0)


# In[ ]:




