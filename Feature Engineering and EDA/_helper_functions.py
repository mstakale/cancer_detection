#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import os.path, sys
from skimage.io import imread, imshow
import cv2
from skimage import data, color, feature , exposure
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import glob
import os


# # Function To Resize Images:
# 
#         Takes input of image path and image final size

# In[3]:


def resize_aspect_fit(path,final_size):
    dirs = os.listdir(path)
    #final_size = 180
    resized_img = []
    for item in dirs:
        if item == '.DS_Store':
             continue
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            size = im.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x*ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
            #name,ext = item.split('.')
            #new_im.save(save_path + name + '.jpg', 'JPEG', quality=100)
            
            resized_img.append(np.asarray(new_im))
            
    return np.array(resized_img)


# # Funcation To Extract SURF features
# 
#         Take input of image array

# In[ ]:


def extract_surf_feat(path):
    surf_img = []
    img_path = glob.glob(path)
    surf = cv2.xfeatures2d.SURF_create(20)
    for img in img_path:
        img1 = cv2.imread(img)
        key, des = surf.detectAndCompute(img1, None)
        result_img = cv2.drawKeypoints(img1,key, None)
        img_con = Image.fromarray(result_img.astype(np.uint8))
        resize_im = img_con.resize((150, 150), Image.ANTIALIAS)
        final_im = np.array(resize_im)
        surf_img.append(final_im)
        
    return np.array(surf_img)

# # Functions to adjust HUE of image

# In[4]:


def hueShift(img, amount):
    arr = np.array(img)
    hsv = rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = hsv_to_rgb(hsv)
    return Image.fromarray(rgb, 'RGB')


def hsv_to_rgb(hsv):
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def rgb_to_hsv(rgb):
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


# # Convert hue of image and retrun image with HUE-SIFT features

# In[ ]:


def get_hue_sift(img_arr):
    
    hue_sift = []
    sift = cv2.xfeatures2d.SIFT_create(20)
    
    for img in range(img_arr.shape[0]):
        for amount in (1, 60):
            im_hue = hueShift(img_arr[img], amount/360)
        hue_img = np.array(im_hue)
        gray= cv2.cvtColor(hue_img,cv2.COLOR_HSV2RGB)
        kp = sift.detect(hue_img,None)
        sift_img=cv2.drawKeypoints(hue_img, kp, outImage = None)
        hue_sift.append(np.asarray(sift_img))
        
    return np.array(hue_sift)


# # Functions To extract LBP features

# In[ ]:


def _calc_texture_gradient(img):
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for color_channel in (0, 1, 2):
        ret[:, :, color_channel] = feature.local_binary_pattern(img[:, :, color_channel], 8, 1.0)
    return ret 


# In[ ]:


def get_lbp_feat(img_arr):
    
    lbp_feat = []
    
    for img in img_arr:
        lbp_img = _calc_texture_gradient(img)
        lbp_feat.append(lbp_img)
    return np.array(lbp_feat)


# # Function To Get HOG Features

# In[5]:


def get_hog_feat(image_arr):
    hog_img = []
    hog_feat = []
    for img in image_arr:
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(10,10),
                       cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_img.append(hog_image_rescaled)
        hog_feat.append(fd)            
    return np.array(hog_img), np.array(hog_feat)


# In[ ]:




