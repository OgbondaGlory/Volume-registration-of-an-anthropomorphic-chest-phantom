#!/usr/bin/env python3
# coding: utf-8

# # 3D-3D Medical Imaging Segmentation, allignment and Deformable Registration of the Human Chest Using AI

# In[1]:
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage
from SimpleITK import DemonsRegistrationFilter
from torch.utils.data import Dataset, DataLoader
from voxelmorph.torch.networks import VxmDense
from torch.optim import Adam
from scipy.ndimage import map_coordinates
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


from utils import *
from RigidDeformation import perform_rigid_registration
from DeformableBsplineRegistration import perform_deformable_bspline_registration
from DemonsRegistration import apply_demons_algorithm, resample_moving_image


# !pip install numpy-stl
# In[2]:
# Loading the Data
if len(sys.argv) != 3:
    raise IOError("Invalid cmd line,\nUsage: " + sys.argv[0] + "   DICOM_PATH   OUTPUT_PATH")

data_path = r"Phantom_CT_Scan"
moving_image_path = r"Phantom_CT_Scan"
fixed_image_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(output_path):
    os.mkdir(output_path)


# List all files in the directory
files = os.listdir(data_path)
print(files)

# Load the DICOM images
ct_image = load_dicom_series(data_path)


# In[7]:


# Perform lung segmentation
lung_mask = segment_lung(ct_image)

# Convert the SimpleITK image to a numpy array for visualization
# lung_mask_array = sitk.GetArrayFromImage(lung_mask)

# Display segmented lungs in 3D
display_images(lung_mask, "Segmented Lungs")


# In[8]:


# Perform bone segmentation
bone_mask = segment_bones(ct_image)

## Display the bone segmentation in 3D and slices
display_images(bone_mask, "Segmented Bones")

# Save segmented lung and bone images
sitk.WriteImage(lung_mask, os.path.join(output_path, "lung_segmentation.mhd"))
sitk.WriteImage(bone_mask, os.path.join(output_path, "bone_segmentation.mhd"))



# _______________________

# ## Initial Alignment

# This section of the code provided performs an initial alignment of the images, which is a critical step in the registration process. However, it doesn't directly use organ masks for this process. Instead, it performs the alignment based on the entire image.
#
# In our code, the initial alignment is done using the CenteredTransformInitializer function. This function aligns the centers of the fixed and moving images in the geometrical sense, which can be thought of as a basic form of rigid registration (translating and rotating the source image to align it with the target image).
#
# This initial alignment is crucial because it provides a good starting point for the subsequent optimization process, which refines the transformation parameters to achieve a better alignment. The metric for this optimization is the mean squares difference between the intensities of the fixed and moving images.
#

# In[9]:


# In[10]:

# Paths to the DICOM directories

# Load the DICOM images
fixed_image = load_dicom_series(fixed_image_path)
moving_image = load_dicom_series(moving_image_path)

reader = sitk.ImageFileReader()
writer = sitk.ImageFileWriter()

if not os.path.exists(output_path + "/target.mha"):
    writer.SetFileName(output_path + "/target.mha")
    writer.Execute(fixed_image)

if not os.path.exists(output_path + "/source.mha"):
    writer.SetFileName(output_path + "/source.mha")
    writer.Execute(moving_image)

# Display the images with appropriate titles
display_images(fixed_image, "Fixed Image patient's CT scan")
display_images(moving_image, "Moving Image CT scan of the phantom")



# In[11]:
# Call the rigid registration function
final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Transformation")
display_images(resampled_moving_image, "Resampled Moving Image after Transformation")
# Generate checkerboard for Rigid Registration
# resampled_moving_image = sitk.ReadImage(output_path + "/rigid_registration.mha")
checker_image = generate_checkerboard(fixed_image, resampled_moving_image)
# Display the checkerboard image for B-Spline deformation
display_images(checker_image, "Checkerboard for Rigid Registration")


# In[]:
#Extracting the ISO Surfaces for Rigid Registration

# Set paths
output_resampled_image_path = os.path.join(output_path, "rigid_registration.mha")
output_iso_surface_file_path = os.path.join(output_path, "iso_surface_rigid.stl")

# Load the image
resampled_moving_image = sitk.ReadImage(output_resampled_image_path)

# Extract the iso-surface
verts, faces = extract_iso_surface(resampled_moving_image, level=0.5, smooth=0.0) # you may need to adjust the level and smooth values

# Save the iso-surface as STL
save_iso_surface(verts, faces, output_iso_surface_file_path)


# In[13]:

# ______________________

# ## Deformable Registration
#

# This section of our code includes a deformable registration phase using B-spline Free-Form Deformation. This approach treats the transformation as a smooth displacement field where control points influence the deformation of a region around them. In this case, I'll use the B-spline transform to create the displacement field and optimize it based on the Mean Squares metric.
#
# The general approach to deformable registration in SimpleITK involves setting up an ImageRegistrationMethod with a BSplineTransformInitializer. The parameters of this initial B-spline transform are then optimized during the registration process to achieve the best alignment.

# Call the deformable B-spline registration function
final_deformable_transform, resampled_moving_image_deformable = perform_deformable_bspline_registration(
    fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Transformation")
display_images(resampled_moving_image, "Resampled Moving Image after Transformation")
# Generate checkerboard for B-Spline deformation
checker_image_deformable = generate_checkerboard(fixed_image, resampled_moving_image_deformable)
# Display the checkerboard image for B-Spline deformation
display_images(checker_image_deformable, "Checkerboard for B-Spline deformation")

# In[]:
#Extracting the ISO Surfaces for B-spline

# Set paths
output_resampled_image_path = os.path.join(output_path, "deformable_registration.mha")
output_iso_surface_file_path = os.path.join(output_path, "iso_surface_deformable.stl")

# Load the image
resampled_moving_image_deformable = sitk.ReadImage(output_resampled_image_path)

# Extract the iso-surface
verts, faces = extract_iso_surface(resampled_moving_image_deformable, level=0.5, smooth=0.0) # you may need to adjust the level and smooth values

# Save the iso-surface as STL
save_iso_surface(verts, faces, output_iso_surface_file_path)
# ______________________________________


# In[16]:
#  ## Apply Demons Algorithim.
# Call the Demons algorithm function
demons_transform = apply_demons_algorithm(fixed_image, resampled_moving_image)

# Check if the resampled moving image already exists. If not, compute it.
output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")
if os.path.exists(output_resampled_image_path):
    resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)
else:
    resampled_moving_image_demons = resample_moving_image(fixed_image, moving_image, demons_transform)
    writer.SetFileName(output_resampled_image_path)
    writer.Execute(resampled_moving_image_demons)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Demons Transformation")
display_images(resampled_moving_image_demons, "Resampled Moving Image after Demons Transformation")
# Generate checkerboard for Demons registration
# resampled_moving_image_demons = sitk.ReadImage(output_path + "/resampled_moving_image_demons.mha")
checker_image_demons = generate_checkerboard(fixed_image, resampled_moving_image_demons)
# Display the checkerboard image for Demons registration
display_images(checker_image_demons, "Checkerboard for Demons registration")


# In[]:
#Extracting the ISO Surfaces for Demons Algorithim

# Set paths
output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")
output_iso_surface_file_path = os.path.join(output_path, "iso_surface_demons.stl")

# Load the image
resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)

# Extract the iso-surface
verts, faces = extract_iso_surface(resampled_moving_image_demons, level=0.5, smooth=0.0) # you may need to adjust the level and smooth values

# Save the iso-surface as STL
save_iso_surface(verts, faces, output_iso_surface_file_path)



# In[18]:#Applying CNNS

# Paths to your directories
# output_path = "/model"
phantom_directory_path = r"Phantom_CT_Scan"
patient_directory_path = r"Patient_CT_Scan"

# Training flag
do_training = True  # Change to False if you want to skip training

# Helper functions
# Define your CTScanDataset, CTScanPairDataset, load_data_for_training, preprocess, apply_displacement_field, and display_images functions here
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
# In[19]:

#Assuming that you want to pair every combination of images in the dataset, not just consecutive ones
# from itertools import combinations

# def load_data_for_training(directory_path):
#     dataset = CTScanDataset(directory_path)
#     pairs = [pair for pair in combinations(dataset, 2)]
#     return CTScanPairDataset(pairs)


# random pairing
# import random

# def load_data_for_training(directory_path):
#     dataset = CTScanDataset(directory_path)
#     random.shuffle(dataset) # Shuffle the images
#     pairs = [(dataset[i], dataset[i + 1]) for i in range(len(dataset) - 1)]
#     return CTScanPairDataset(pairs)

# In[20]:
def compute_similarity(image1, image2, method='correlation'):
    image1_np = sitk.GetArrayFromImage(image1)
    image2_np = sitk.GetArrayFromImage(image2)

    if method == 'correlation':
        similarity = np.corrcoef(image1_np.flat, image2_np.flat)[0, 1] # correlation coefficient
    elif method == 'ssim':
        similarity = ssim(image1_np, image2_np)
    elif method == 'mse':
        similarity = -mean_squared_error(image1_np, image2_np) # MSE is a distance, so negate it to make it a similarity
    else:
        raise ValueError(f'Unknown method: {method}')

    return similarity


def load_data_for_training(directory_path, method='correlation'):
    # Load the images
    dataset = CTScanDataset(directory_path)

    # Compute the similarity between every pair of images
    similarities = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(i, len(dataset)):
            if i != j:
                similarity = compute_similarity(dataset[i], dataset[j], method=method)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

    # Pair each image with the most similar other image
    pairs = []
    for i in range(len(dataset)):
        j = np.argmax(similarities[i])
        pairs.append((dataset[i], dataset[j]))

    return CTScanPairDataset(pairs)



def preprocess(image_array):
    # Ensure the output is (channel, depth, height, width)
    image_array = np.expand_dims(image_array, axis=0)
    image_tensor = torch.from_numpy(image_array)
    return image_tensor

def apply_displacement_field(moving_image_array, displacement_field):
    coords = np.mgrid[0:moving_image_array.shape[0], 0:moving_image_array.shape[1], 0:moving_image_array.shape[2]]
    coords += displacement_field
    warped_moving_image_array = map_coordinates(moving_image_array, coords, order=3)
    return warped_moving_image_array

if os.path.exists(output_path + "/model.pth") and not do_training:
    # Load the pre-trained model
    model = VxmDense(inshape=(256, 256, 256), nb_unet_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]])
    model.load_state_dict(torch.load(os.path.join(output_path, "model.pth")))

else:
    # Initialize the model and optimizer
    model = VxmDense(inshape=(256, 256, 256), nb_unet_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]])
    optimizer = Adam(model.parameters())

    # Load the phantom and patient CT scans
    phantom_dataset = CTScanDataset(phantom_directory_path)
    patient_dataset = CTScanDataset(patient_directory_path)

    # Pair up the phantom and patient scans
    pairs = list(zip(phantom_dataset, patient_dataset))
    paired_dataset = CTScanPairDataset(pairs)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        for i, (fixed_image, moving_image) in enumerate(paired_dataset):
            optimizer.zero_grad()
            y_pred, _ = model(moving_image.float().unsqueeze(0), fixed_image.float().unsqueeze(0))
            loss = torch.nn.MSELoss()(y_pred, fixed_image.float().unsqueeze(0))
            loss.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))

# Apply the model to a pair of images for visualization
fixed_image_tensor, moving_image_tensor = paired_dataset[0]  # Change index if necessary

# Compute the displacement field
displacement_field, _ = model([fixed_image_tensor.float().unsqueeze(0), moving_image_tensor.float().unsqueeze(0)])

# Convert displacement field back to numpy
displacement_field = displacement_field.detach().numpy()

# Apply displacement field to moving image
warped_moving_image_array = apply_displacement_field(moving_image_tensor.numpy(), displacement_field)

#Extracting ISO Surfaces for CNNS
# Set paths
output_warped_image_path = os.path.join(output_path, "warped_image.mha")
output_iso_surface_file_path = os.path.join(output_path, "iso_surface_warped.stl")

# Save the warped image
sitk.WriteImage(sitk.GetImageFromArray(warped_moving_image_array.squeeze()), output_warped_image_path)

# Load the image
warped_image = sitk.ReadImage(output_warped_image_path)

# Extract the iso-surface
verts, faces = extract_iso_surface(warped_image, level=0.5, smooth=0.0) # you may need to adjust the level and smooth values

# Save the iso-surface as STL
save_iso_surface(verts, faces, output_iso_surface_file_path)

# Display the images after transformation
display_images(fixed_image_tensor.numpy().squeeze())
display_images(warped_moving_image_array.squeeze())
