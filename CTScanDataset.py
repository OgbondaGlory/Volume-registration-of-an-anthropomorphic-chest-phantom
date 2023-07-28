# In[1]:
import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

# In[2]:
# Helper functions
# Define your CTScanDataset, CTScanPairDataset
class CTScanDataset(Dataset):
    def __init__(self, directory_path):
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.files[idx])
        image_array = sitk.GetArrayFromImage(image)
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        return image_array

class CTScanPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fixed_image, moving_image = self.pairs[idx]
        fixed_image = self.preprocess(fixed_image)
        moving_image = self.preprocess(moving_image)
        return fixed_image, moving_image

    def preprocess(self, image_array):
        # Resize to (256, 256, 256)
        image_array = scipy.ndimage.zoom(image_array, (256/image_array.shape[0], 256/image_array.shape[1], 256/image_array.shape[2]))
        # Ensure the output is (channel, depth, height, width)
        image_array = np.expand_dims(image_array, axis=0)
        image_tensor = torch.from_numpy(image_array)
        return image_tensor
