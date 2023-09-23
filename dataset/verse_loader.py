# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证
from scipy import ndimage
import glob
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import random

def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    axis = random.sample(range(0, 3), 2)
    image = np.rot90(image, k, axes=(axis[0], axis[1]))  # rot along z axis
    label = np.rot90(label, k, axes=(axis[0], axis[1]))

    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-15, 15)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(image, angle, axes=rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    label = ndimage.interpolation.rotate(label, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)

    return image, label


# z, y, x     0, 1, 2
def rot_from_y_x(image, label):
    # k = np.random.randint(0, 4)
    image = np.rot90(image, 2, axes=(1, 2))  # rot along z axis
    label = np.rot90(label, 2, axes=(1, 2))

    return image, label


def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label




class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向


        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        Delta_z = np.int((z_max - z_min) / 3)  # 3
        Delta_y = np.int((y_max - y_min) / 4)  # 8
        Delta_x = np.int((x_max - x_min) / 4)  # 8

        # random number of x, y, z
        # z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
        
        thre = z_min + Delta_z + np.int(self.output_size[0] / 2)
        if z_middle > thre:          # 此时z_middle + Delta_z < z_max
            delta_Z = z_middle - z_min - np.int(self.output_size[0] / 4)                         # 正常 np.int(self.output_size[0] / 2)，此时再大点，保证可以超出现有的范围
            z_random = random.randint(z_middle - delta_Z, z_middle + delta_Z)
        else:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)

         # padding
        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

        label = np.round(label)

         # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(np.float)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()
        
        sample = {'image': image, 'label': label.long()}
        return sample


def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image

class PretrainDataset(Dataset):
    def __init__(self, num_classes,sample_list,data_list, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.data_list = data_list
        self.cache = cache
        self.num_classes=num_classes
        self.sample_list=sample_list
        
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(data_list)), total=len(data_list)):
                d  = self.read_data(data_list[i])
                self.cache_data.append(d)

    def read_data(self,data_list):
        data_path =data_list
        
        data = np.load(data_path)
       
    #   image_data = data["data"][0:1].astype(np.float32)
        
        #seg_data = data["data"][1:2].astype(np.float32)

        image_data = np.squeeze(data["data"][0:1], axis=0).astype(np.float32)
        seg_data = np.squeeze(data["data"][1:2], axis=0).astype(np.float32)
        


        
        seg_data [seg_data  < 0.5] = 0.0  # maybe some voxels is a minus value
        seg_data [seg_data  > 25.5] = 0.0

        return {
            "image": image_data,
            "label": seg_data

        } 


    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            try:
                image = self.read_data(self.data_list[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"数据读取出现问题，{self.data_list[i]}\n")
                if i != len(self.data_list)-1:
                    return self.__getitem__(i+1)
                else :
                    return self.__getitem__(i-1)
        if self.transform is not None :
            image = self.transform(image)
            image['label']=np.expand_dims(np.array(image['label']),axis=0) 

            

        image['case_name'] = self.sample_list[i]
        
        return image

    def __len__(self):
        return len(self.data_list)

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

class Args:
    def __init__(self) -> None:
        self.workers=8
        self.fold=0
        self.batch_size=2
        
#autodl-tmp/UniSeg/Upstream/nnUNet_preprocessed/Task037_VerSe20binary/nnUNetData_plans_v2.1_stage0
def get_loader_verse(data_dir,list_dir,batch_size=1,fold=0, num_workers=8):
    
    train_list = open(os.path.join(list_dir, 'train.txt')).readlines()
    train_list = [item.strip() for item in train_list]
    
   
    val_list = open(os.path.join(list_dir, 'val.txt')).readlines()
    val_list = [item.strip() for item in val_list]
    
    test_list = open(os.path.join(list_dir, 'test.txt')).readlines()
    test_list = [item.strip() for item in test_list]
    
    
    data_files=glob.glob(os.path.join(data_dir+'/*.npz'))
    
    
    train_size = len(train_list)
    val_size = len(val_list)
    test_size=len(test_list)
    
    train_files=[i for i in data_files if  i.split('.npz')[0].split('/')[-1] in train_list]
    val_files=[i for i in data_files if  i.split('.npz')[0].split('/')[-1] in val_list]
    test_files=[i for i in data_files if  i.split('.npz')[0].split('/')[-1] in test_list]
    
    print(f"train_size is {len(train_files)}, val_size is {len(val_files)}, test_size is {len(test_files)}")
    
    img_size=[128,160,96]
    train_ds = PretrainDataset(int (26),train_list,train_files,
                               transform=transforms.Compose([RandomGenerator(output_size=img_size, mode = 'train')]))

    val_ds = PretrainDataset(int (26),val_list,val_files,
                                     transform=transforms.Compose(
                                   [RandomGenerator(output_size=img_size, mode = 'val')]))
    

    test_ds = PretrainDataset(int (26),test_list,test_files,
                              transform=transforms.Compose(
                                   [RandomGenerator(output_size=img_size, mode = 'test')]))

    loader = [train_ds, val_ds, test_ds]

    return loader
