import numpy as np
import torch
import os

class AddPadding(object):
  def __init__(self, output_size):
    self.output_size = output_size

  def resize_image_by_padding(self,image,new_shape,pad_value=0):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res
  
  def __call__(self, sample):
    sample['data']=self.resize_image_by_padding(sample['data'],new_shape=self.output_size)
    if("gt" in sample.keys()):
      sample['gt']=self.resize_image_by_padding(sample['gt'],new_shape=self.output_size)
    return sample

class CenterCrop(object):
  def __init__(self, output_size):
    self.output_size = output_size

  def center_crop_2D_image(self,img,crop_size):
    if(all(np.array(img.shape)<=crop_size)):
      return img
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
      center_crop = [int(crop_size)] * len(img.shape)
    else:
      center_crop = crop_size
      assert len(center_crop) == len(
          img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]
  
  def __call__(self, sample):
    sample['data']=self.center_crop_2D_image(sample['data'],crop_size=self.output_size)
    if("gt" in sample.keys()):
      sample['gt']=self.center_crop_2D_image(sample['gt'],crop_size=self.output_size)
    return sample

class OneHot(object):
  def one_hot(self,seg,num_classes=4):
    return np.eye(num_classes)[seg.astype(int)].transpose(2,0,1)
  def __call__(self, sample):
    sample['gt']=self.one_hot(sample['gt'])
    return sample

class ToTensor(object):
  def __call__(self, sample):
    sample['data']=torch.from_numpy(sample['data'][None,:,:]).float()
    if("gt" in sample.keys()):
      sample['gt']=torch.from_numpy(sample['gt']).float()
    return sample

#TODO SpatialTransform, MirrorTransform
#TODO Subclasses

class ACDCDataLoader():
  def __init__(self,root_dir,patient_ids,has_gt,transform,batch_size):
    self.patient_ids=patient_ids
    self.batch_size=batch_size
    self.patient_loaders=[]
    for id in patient_ids:
      self.patient_loaders.append(torch.utils.data.DataLoader(
          ACDCPatient(root_dir,id,has_gt=has_gt,transform=transform),
          batch_size=batch_size,shuffle=False,num_workers=0
      ))
    self.counter_id=0
    self.counter_iter=None

  def __iter__(self):
    self.counter_iter=0
    return self
  
  def __next__(self):
    if(self.counter_iter is None):
      self.counter_iter=0
    if(self.counter_iter==len(self)):
      raise StopIteration
    loader=self.patient_loaders[self.counter_id]
    self.counter_id+=1
    self.counter_iter+=1
    if(self.counter_id%len(self)==0):
      self.counter_id=0
    return loader

  def __len__(self):
    return len(self.patient_ids)

  def current_id(self):
    return self.patient_ids[self.counter_id]

class ACDCPatient(torch.utils.data.Dataset):
  def __init__(self,root_dir,patient_id,has_gt,transform=None):
    self.root_dir=root_dir
    self.has_gt=has_gt
    self.id=patient_id
    self.info=np.load("preprocessed/patient_info.npy",allow_pickle=True).item()[patient_id]
    self.transform=transform

  def __len__(self):
    return self.info["shape_ED"][2]+self.info["shape_ES"][2]
  
  def __getitem__(self, slice_id):
    is_es=slice_id>=len(self)//2
    slice_id=slice_id-len(self)//2 if is_es else slice_id
    data=np.load(os.path.join(self.root_dir,"patient{:03d}.npy".format(self.id)))
    sample={
        "data": data[0,slice_id] if not is_es else data[1,slice_id]
    }
    if(self.has_gt):
      sample["gt"]=data[2,slice_id] if not is_es else data[3,slice_id]
    if self.transform:
      sample = self.transform(sample)
    return sample