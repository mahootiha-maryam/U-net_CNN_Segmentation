# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:44:56 2021

@author: Asus
"""

#1.import modules

import os
import glob
import keras
import random
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as k
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

#2.define train and test path

train_image_path='D:\images\TIF\input-train'
train_mask_path='D:\images\TIF\mask-train'
test_image_path='D:\images\TIF\input-test'
test_mask_path='D:\images\TIF\mask-test'


#3.initialize images and masks size

img_height, img_width, img_channels= 256, 256, 3


#4.define preprocessing

train_mask_list= sorted(next(os.walk(train_mask_path))[2])
test_mask_list= sorted(next(os.walk(test_mask_path))[2])

def data_preprocessing_train():
    init_image=np.zeros((len(train_mask_list),288,384,3),dtype=np.uint8)
    init_mask=np.zeros((len(train_mask_list),288,384),dtype=bool)
    train_x=np.zeros((len(train_mask_list),img_height, img_width, img_channels),dtype=np.uint8)
    train_y=np.zeros((len(train_mask_list),img_height, img_width, 1),dtype=bool)
    
    n=0
    
    for mask_path in glob.glob('{}/*.tif'.format(train_mask_path)):
        base=os.path.basename(mask_path)
        image_id, ext= os.path.splitext(base)
        image_path='{}/{}.tif'.format(train_image_path, image_id)
        mask=imread(mask_path)
        image=imread(image_path)
        
        y_coord, x_coord= np.where(mask==255)
        y_min=min(y_coord)
        y_max=max(y_coord)
        x_min=min(x_coord)
        x_max=max(x_coord)
        
        
        cropped_image=image[y_min:y_max, x_min:x_max]
        cropped_mask=mask[y_min:y_max, x_min:x_max]
        
        train_x[n]=resize(cropped_image[:,:,:img_channels],
                          (img_height, img_width, img_channels),
                          mode='constant', 
                          anti_aliasing= True, 
                          preserve_range=True
                          )
        train_y[n]=np.expand_dims(resize(cropped_mask,
                          (img_height, img_width),
                          mode='constant', 
                          anti_aliasing= True, 
                          preserve_range=True
                          ),axis=-1)
        
        init_image[n]=image
        init_mask[n]=mask
        
        n+=1
        
    return train_x, train_y, init_image, init_mask 

train_input, train_mask, init_image, init_mask=data_preprocessing_train()       


def data_preprocessing_test():
    test_x=np.zeros((len(test_mask_list),img_height, img_width, img_channels),dtype=np.uint8)
    test_y=np.zeros((len(test_mask_list),img_height, img_width, 1),dtype=bool)
    
    n=0
    
    for mask_path in glob.glob('{}/*.tif'.format(test_mask_path)):
        base=os.path.basename(mask_path)
        image_id, ext= os.path.splitext(base)
        image_path='{}/{}.tif'.format(test_image_path, image_id)
        mask=imread(mask_path)
        image=imread(image_path)
        
        y_coord, x_coord= np.where(mask==255)
        y_min=min(y_coord)
        y_max=max(y_coord)
        x_min=min(x_coord)
        x_max=max(x_coord)
        
        
        cropped_image=image[y_min:y_max, x_min:x_max]
        cropped_mask=mask[y_min:y_max, x_min:x_max]
        
        test_x[n]=resize(cropped_image[:,:,:img_channels],
                          (img_height, img_width, img_channels),
                          mode='constant', 
                          anti_aliasing= True, 
                          preserve_range=True
                          )
        test_y[n]=np.expand_dims(resize(cropped_mask,
                          (img_height, img_width),
                          mode='constant', 
                          anti_aliasing= True, 
                          preserve_range=True
                          ),axis=-1)

        n+=1
        
    return test_x, test_y

test_input, test_mask=data_preprocessing_test()
    
    #4.1 show the results for preprocessing



print('region_of_interest_image')
imshow(train_input[0])
plt.show()
 
print('region_of_interest_mask')
imshow(train_mask[0])
plt.show()

print('original_image')
imshow(init_image[0])
plt.show()

print('original_mask')
imshow(np.squeeze(init_mask[0]))
plt.show()

rows=1
columns=4
figure=plt.figure(figsize=(15,15))
image_list= [init_image[0], init_mask[0], train_input[0], train_mask[0]]

for i in range(1, rows*columns +1):
    image= image_list[i-1]
    sub_plot_image=figure.add_subplot(rows, columns, i)
    sub_plot_image.imshow(np.squeeze(image))
plt.show()


#5.implementation of unet model for segmentation

def unet_segmentation(input_size=(img_height, img_width, img_channels)):
    inputs=input(input_size)
    n=Lambda(lambda x:x/255)(inputs)

    
    c1=Conv2D(16, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(n)
    c1=Dropout(0.1)(c1)
    c1=Conv2D(16, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(c1)
    p1=MaxPooling2D((2,2))(c1)



    c2=Conv2D(32, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(p1)
    c2=Dropout(0.1)(c2)
    c2=Conv2D(32, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(c2)
    p2=MaxPooling2D((2,2))(c2)
    
    
    c3=Conv2D(64, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(p2)
    c3=Dropout(0.2)(c3)
    c3=Conv2D(64, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(c3)
    p3=MaxPooling2D((2,2))(c3)
    
    
    
    c4=Conv2D(128, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(p3)
    c4=Dropout(0.2)(c4)
    c4=Conv2D(128, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(c4)
    p4=MaxPooling2D((2,2))(c4)
    
    
    
    c5=Conv2D(256, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(p4)
    c5=Dropout(0.3)(c5)
    c5=Conv2D(256, (3,3), activation='elu', 
              kernel_initializer='he_normal',
              padding='same')(c5)
    
    
    #unconvolutional blocks
    
    u6=Conv2DTranspose(128, (2,2), strides=(2,2),
                       padding='same')(c5)
    u6=concatenate([u6,c4])
    c6=Conv2D(128, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(u6)
    c6=Dropout(0.2, c6)
    c6=Conv2D(128, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(c6)
    
    
    u7=Conv2DTranspose(64, (2,2), strides=(2,2),
                       padding='same')(c6)
    u7=concatenate([u7,c3])
    c7=Conv2D(64, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(u7)
    c7=Dropout(0.2, c7)
    c7=Conv2D(64, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(c7)
    
    
    u8=Conv2DTranspose(32, (2,2), strides=(2,2),
                       padding='same')(c7)
    u8=concatenate([u8,c2])
    c8=Conv2D(32, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(u8)
    c8=Dropout(0.1, c8)
    c8=Conv2D(32, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(c8)
    
    
    u9=Conv2DTranspose(16, (2,2), strides=(2,2),
                       padding='same')(c8)
    u9=concatenate([u9,c1],axis=3)
    c9=Conv2D(16, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(u9)
    c9=Dropout(0.1, c9)
    c9=Conv2D(16, (3,3), activation='elu',
              kernel_initializer='he_normal',
              padding='same')(c9)
    
    
    
    output=Conv2D(1,(1,1),activation='sigmoid')(c9)
    
    
    model=Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[Mean_IOU_Evaluator])
    model.summary()
    return model

#6.define unet model evaluator

def Mean_IOU_Evaluator(y_true, y_pred):
    prec=[]
    for t in np.arange(0.5,1,0.05):
        
        y_pred_=tf.to_int32((y_pred>t))
        score, up_opt=tf.metrics.mean_iou(y_true,y_pred_,2)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score=tf.identity(score)
            prec.append(score)
    return k.mean(k.stack(prec), axis=0)

model=unet_segmentation()

#7.show the results per epoch

class loss_history(keras.callbacks.Callback):
    def __init__(self, x=4):
        self.x=x
    def on_epoch_begin(self, epoch,logs={}):
        imshow(train_input[self.x])
        plt.show()
        
        imshow(np.squeeze(train_mask[self.x]))
        plt.show()
        
        pred_train=self.model.predict(np.expand_dims(train_input[self.x]),axis=0)
        imshow(np.squeeze(pred_train[0]))
        plt.show()

imageset='PCC'
backbone='unet'
version='v1.0'

model_h5='model-{imageset}-{backbone}-{version}'.format(
    imageset=imageset,backbone=backbone, version=version)

model_h5_checkpoint='{model_h5}.checkpoint'.format(model_h5=model_h5)
earlystopper=EarlyStopping(patience=7, verbose=1)
checkpointer=ModelCheckpoint(model_h5_checkpoint, verbose=1, save_best_only= True)



#8.train unet model using training samples

results=model.fit(train_input, train_mask,
                  validation_split=0.1,
                  batch_size=2,
                  epochs=50,
                  callbacks=[earlystopper, checkpointer, loss_history()])

#9.unet model evaluation using test

preds_train=model.predict(train_input, verbose=1)
preds_train_t=(preds_train>0.5).astype(np.uint8)
preds_test=model.predict(test_input, verbose=1)
preds_test_t=(preds_test>0.5).astype(np.uint8)


#10.show final results

#random show result

ix=random.randint(0, len(train_input)-1)

print(ix)
print('train image')
imshow(train_input[ix])
plt.show()

print('train mask')
imshow(np.squeeze(train_mask[ix]))
plt.show()

print('segmented image')
imshow(np.squeeze(preds_train[ix]))
plt.show()

iix=random.randint(0, 1)
print(iix)

print('test image')
imshow(test_input[iix])
plt.show()

print('test mask')
imshow(np.squeeze(test_mask[iix]))
plt.show()


print('segmented mask')
imshow(np.squeeze(preds_test[iix]))
plt.show()
