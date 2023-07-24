#!/usr/bin/env python3
# coding: utf-8

# # 3D-3D Medical Imaging Segmentation, allignment and Deformable Registration of the Human Chest Using AI

# In[1]:

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from SimpleITK import DemonsRegistrationFilter


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


# In[22]:


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

# In[23]:


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


# In[28]:


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

# In[29]:






# Display the images after transformation
display_images(fixed_image, "Fixed Image after Transformation")
display_images(resampled_moving_image, "Resampled Moving Image after Transformation")


# ______________________

# ## Deformable Registration
#

# This section of our code includes a deformable registration phase using B-spline Free-Form Deformation. This approach treats the transformation as a smooth displacement field where control points influence the deformation of a region around them. In this case, I'll use the B-spline transform to create the displacement field and optimize it based on the Mean Squares metric.
#
# The general approach to deformable registration in SimpleITK involves setting up an ImageRegistrationMethod with a BSplineTransformInitializer. The parameters of this initial B-spline transform are then optimized during the registration process to achieve the best alignment.

# In[ ]:


# # Apply the affine transformation
# affine_registration_method = sitk.ImageRegistrationMethod()
# # affine_registration_method.SetMetricAsMeanSquares()
# affine_registration_method.SetMetricAsCorrelation()
# affine_registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# affine_registration_method.SetInterpolator(sitk.sitkLinear)

# initial_affine_transform = sitk.CenteredTransformInitializer(fixed_image,
#                                                               moving_image,
#                                                               sitk.Euler3DTransform(),
#                                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)

# affine_registration_method.SetInitialTransform(initial_affine_transform, inPlace=False)
# final_affine_transform = affine_registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
#                                                              sitk.Cast(moving_image, sitk.sitkFloat32))


# In[ ]:
if os.path.exists(output_path + "/deformable_transformation.tfm") and os.path.exists(output_path + "/composite_transform.tfm") and os.path.exists(output_path + "/deformable_registration.mha"):
    final_deformable_transform = sitk.ReadTransform(output_path + "/deformable_transformation.mha")
    composite_transform = sitk.ReadTransform(output_path + "/composite_transform.mha")

    reader.SetFileName(output_path + "/deformable_registration.mha")
    resampled_moving_image = reader.Execute()

else:
    # Now set up the deformable registration (B-spline)
    deformable_registration_method = sitk.ImageRegistrationMethod()
    # deformable_registration_method.SetMetricAsMeanSquares()
    deformable_registration_method.SetMetricAsCorrelation()
    deformable_registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    deformable_registration_method.SetInterpolator(sitk.sitkLinear)

    # Initialize the B-spline transform
    transform_domain_physical_dim_size = fixed_image.GetSize()
    transform_domain_mesh_size = [size//8 for size in transform_domain_physical_dim_size] # Arbitrary mesh size, you might need to adjust this
    initial_deformable_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                                    transformDomainMeshSize=transform_domain_mesh_size, order=3)

    deformable_registration_method.SetInitialTransform(initial_deformable_transform)
    final_deformable_transform = deformable_registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                                        sitk.Cast(resampled_moving_image, sitk.sitkFloat32))


# In[ ]:


    # Combine the affine and deformable transforms
    composite_transform = sitk.Transform(fixed_image.GetDimension(), sitk.sitkComposite)
    composite_transform.AddTransform(final_transform_v1)
    composite_transform.AddTransform(final_deformable_transform)

    sitk.WriteTransform(final_deformable_transform, output_path + "/deformable_transformation.tfm")
    sitk.WriteTransform(composite_transform, output_path + "/composite_transform.tfm")

    # Resample the moving image onto the fixed image's grid using the composite transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(composite_transform)
    resampled_moving_image = resampler.Execute(moving_image)

    writer.SetFileName(output_path + "/deformable_registration.mha")
    writer.Execute(resampled_moving_image)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Transformation")
display_images(resampled_moving_image, "Resampled Moving Image after Transformation")



# ______________________________________

#  ## Apply Demons Algorithim.

# In[ ]:
# Import the necessary filter

# Configure and run the Demons Registration
def apply_demons_algorithm(fixed_image, moving_image, iterations=50):
    demons_filter = DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)
    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    return demons_transform

# Set the path for the output demons transformation
output_demons_transform_path = os.path.join(output_path, "demons_transformation.tfm")

# Check if the transform already exists. If not, compute it.
if os.path.exists(output_demons_transform_path):
    demons_transform = sitk.ReadTransform(output_demons_transform_path)
else:
    demons_transform = apply_demons_algorithm(fixed_image, resampled_moving_image)
    sitk.WriteTransform(demons_transform, output_demons_transform_path)

# Resample the moving image onto the fixed image's grid using the demons transform
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetTransform(demons_transform)
resampled_moving_image_demons = resampler.Execute(moving_image)

# Save the resampled image
output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")
writer.SetFileName(output_resampled_image_path)
writer.Execute(resampled_moving_image_demons)

# Display the images after transformation
display_images(fixed_image, "Fixed Image after Demons Transformation")
display_images(resampled_moving_image_demons, "Resampled Moving Image after Demons Transformation")
