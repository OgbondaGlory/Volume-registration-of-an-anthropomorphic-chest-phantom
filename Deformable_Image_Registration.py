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
from skimage.measure import marching_cubes_lewiner
from stl import mesh
# !pip install numpy-stl
# In[2]:



def segment_lung(image):
    # Apply a threshold to separate lung pixels from others
    thresh_filter = sitk.ThresholdImageFilter()
    thresh_filter.SetLower(-1000)
    thresh_filter.SetUpper(-400)
    thresh_img = thresh_filter.Execute(image)

    # Apply morphological opening
    morph_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    morph_img = morph_filter.Execute(thresh_img)

    # Apply Connected Component Labeling
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(morph_img)

    # Get the two largest components (assumed to be the lungs)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_img = relabel_filter.Execute(cc_img)

    # Get labels array
    labels_array = sitk.GetArrayFromImage(relabel_img)

    # Count the number of pixels in each region and sort by size
    unique, counts = np.unique(labels_array, return_counts=True)
    sorted_counts = sorted(zip(counts, unique), reverse=True)

    # Keep the two largest components (excluding background)
    lung_mask = np.isin(labels_array, [label for _, label in sorted_counts[1:3]])

    # Convert to SimpleITK image for further processing
    lung_mask_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
    lung_mask_sitk.CopyInformation(image)

    return lung_mask_sitk


# In[3]:


def segment_bones(image):
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Use thresholding to identify the bones
    # Bones have higher HU than soft tissues or air
    # This range might need adjustment depending on the specific scan
    bone_threshold = 300
    bone_mask = image_array > bone_threshold

    # Convert to SimpleITK image for further processing
    bone_mask_sitk = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
    bone_mask_sitk.CopyInformation(image)

    return bone_mask_sitk


# In[4]:


def load_dicom_series(directory_path):
    reader = sitk.ImageSeriesReader()
#     print(directory_path)
#     dicom_names = files = os.listdir(directory_path).sort()
#     reader.SetFileNames(dicom_names)
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
#     print(dicom_names)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


# In[5]:


def display_images(image, title, montage_slices=10):
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Normalize to 0-255
    image_array = ((image_array - image_array.min()) * (1/(image_array.max() - image_array.min()) * 255)).astype('uint8')
    # Montage of slices using matplotlib
    fig = plt.figure(figsize=(10, 2))
    fig.suptitle(title + " - Slice View")
    slice_interval = image_array.shape[0] // montage_slices
    for i in range(montage_slices):
        ax = fig.add_subplot(1, montage_slices, i + 1)
        ax.imshow(image_array[i * slice_interval], cmap='gray')
        plt.axis('off')
    plt.show()


# Define a simple callback which allows us to monitor registration progress.
def iteration_callback(filter):
    print('\r{0:.2f}'.format(filter.GetMetricValue()), end='')

# In[6]:


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


# _______________________

# ## Initial Alignment

# This section of the code provided performs an initial alignment of the images, which is a critical step in the registration process. However, it doesn't directly use organ masks for this process. Instead, it performs the alignment based on the entire image.
#
# In our code, the initial alignment is done using the CenteredTransformInitializer function. This function aligns the centers of the fixed and moving images in the geometrical sense, which can be thought of as a basic form of rigid registration (translating and rotating the source image to align it with the target image).
#
# This initial alignment is crucial because it provides a good starting point for the subsequent optimization process, which refines the transformation parameters to achieve a better alignment. The metric for this optimization is the mean squares difference between the intensities of the fixed and moving images.
#

# In[9]:
def generate_checkerboard(fixed_image, moving_image, pattern=(5,5,5)):
    checkerboard_filter = sitk.CheckerBoardImageFilter()
    checkerboard_filter.SetCheckerPattern(pattern)
    checker_image = checkerboard_filter.Execute(fixed_image, moving_image)
    return checker_image

def convert_to_sitk(image_array, original_image):
    # Get the metadata from the original image
    spacing = original_image.GetSpacing()
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()

    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    return sitk_image

def extract_iso_surface(image, level, smooth=0.0):
    # Convert SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Extract iso-surfaces
    verts, faces, _, _ = marching_cubes_lewiner(image_array, level=level, step_size=1, smoothing=smooth, allow_degenerate=False)
    return verts, faces

def save_iso_surface(verts, faces, filename):
    # Create a mesh
    iso_surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    # Populate the mesh with data
    for i, f in enumerate(faces):
        for j in range(3):
            iso_surface_mesh.vectors[i][j] = verts[f[j],:]
    
    # Save the mesh to file
    iso_surface_mesh.save(filename)

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

# Rigid Registration
if os.path.exists(output_path + "/rigid_transformation.tfm") and os.path.exists(output_path + "/rigid_registration.mha"):
    final_transform_v1 = sitk.ReadTransform(output_path + "/rigid_transformation.tfm")

    reader.SetFileName(output_path + "/rigid_registration.mha")
    resampled_moving_image = reader.Execute()

else:
    # Apply the transformation
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    print("********************************************************************************")
    print("* Rigid registration                                                            ")
    print("********************************************************************************")
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

    final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                     sitk.Cast(moving_image, sitk.sitkFloat32))

    sitk.WriteTransform(final_transform_v1, output_path + "/rigid_transformation.tfm")

    # Resample the moving image onto the fixed image's grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform_v1)
    resampled_moving_image = resampler.Execute(moving_image)

    writer.SetFileName(output_path + "/rigid_registration.mha")
    writer.Execute(resampled_moving_image)



# In[12]:

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


# In[14]:
if os.path.exists(output_path + "/deformable_transformation.tfm") and os.path.exists(output_path + "/composite_transform.tfm") and os.path.exists(output_path + "/deformable_registration.mha"):
    final_deformable_transform = sitk.ReadTransform(output_path + "/deformable_transformation.mha")
    composite_transform = sitk.ReadTransform(output_path + "/composite_transform.mha")

    reader.SetFileName(output_path + "/deformable_registration.mha")
    resampled_moving_image_deformable = reader.Execute()

else:
    # Now set up the deformable registration (B-spline)
    deformable_registration_method = sitk.ImageRegistrationMethod()
    # deformable_registration_method.SetMetricAsCorrealtion()
    deformable_registration_method.SetMetricAsCorrelation()
    deformable_registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    deformable_registration_method.SetInterpolator(sitk.sitkLinear)

    # Initialize the B-spline transform
    transform_domain_physical_dim_size = fixed_image.GetSize()
    transform_domain_mesh_size = [size//4 for size in transform_domain_physical_dim_size] # Finer mesh size
    initial_deformable_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                                    transformDomainMeshSize=transform_domain_mesh_size, order=3)

    # Use a multi-resolution strategy
    deformable_registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    deformable_registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    deformable_registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    deformable_registration_method.SetInitialTransform(initial_deformable_transform)

    print("********************************************************************************")
    print("* Deformable registration                                                       ")
    print("********************************************************************************")
    deformable_registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(deformable_registration_method))

    final_deformable_transform = deformable_registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                                        sitk.Cast(resampled_moving_image, sitk.sitkFloat32))


# In[15]:

    # Combine the affine and deformable transforms
    composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
    composite_transform.AddTransform(final_transform_v1)
    composite_transform.AddTransform(final_deformable_transform)


    sitk.WriteTransform(final_deformable_transform, output_path + "/deformable_transformation.tfm")
    # sitk.WriteTransform(composite_transform, output_path + "/composite_transform.tfm")

    # Resample the moving image onto the fixed image's grid using the composite transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(composite_transform)
    resampled_moving_image_deformable = resampler.Execute(resampled_moving_image)

    writer.SetFileName(output_path + "/deformable_registration.mha")
    writer.Execute(resampled_moving_image_deformable)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Transformation")
display_images(resampled_moving_image, "Resampled Moving Image after Transformation")
# Generate checkerboard for B-Spline deformation
# resampled_moving_image_deformable = sitk.ReadImage(output_path + "/deformable_registration.mha")
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

# Import the necessary filter

# Define function to apply the Demons Registration algorithm
def apply_demons_algorithm(fixed_image, moving_image, iterations=100):
    demons_filter = DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)

    print("********************************************************************************")
    print("* Demons registration                                                           ")
    print("********************************************************************************")
    deformable_registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(deformable_registration_method))

    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    return demons_transform

# Define function to resample the moving image with a given transform
def resample_moving_image(fixed_image, moving_image, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    return resampler.Execute(moving_image)

# Set the paths for the output demons transformation and resampled image
output_demons_transform_path = os.path.join(output_path, "demons_transformation.tfm")
output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")

# Check if the transform already exists. If not, compute it.
if os.path.exists(output_demons_transform_path):
    demons_transform = sitk.ReadTransform(output_demons_transform_path)
else:
    demons_transform = apply_demons_algorithm(fixed_image, resampled_moving_image)
    sitk.WriteTransform(demons_transform, output_demons_transform_path)

# Check if the resampled moving image already exists. If not, compute it.
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
