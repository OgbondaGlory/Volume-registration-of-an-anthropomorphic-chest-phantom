# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DemonsRegistration.py

# DemonsRegistration.py

import os
import SimpleITK as sitk

def rescale_image(image, new_size):
    """
    Rescale the given image to a new size.
    
    Parameters:
    - image: SimpleITK image to be rescaled.
    - new_size: Desired output size.
    
    Returns:
    - Resampled image with new size.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(
        [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in zip(image.GetSize(), image.GetSpacing(), new_size)])
    resampler.SetOutputDirection(image.GetDirection())
    return resampler.Execute(image)

def apply_demons_algorithm(fixed_image, moving_image, output_path, mask_name, iterations=100, demons_std=1.0):
    """
    Apply the Demons registration algorithm to align moving_image with fixed_image.
    
    Parameters:
    - fixed_image: Target image for registration.
    - moving_image: Image to be registered.
    - output_path: Directory path for saving the results.
    - mask_name: Identifier for naming saved results.
    - iterations: Number of iterations for the Demons algorithm.
    - demons_std: Standard deviation for smoothing in Demons.
    
    Returns:
    - Tuple containing the demons transform and the resampled moving image.
    """
    print("Fixed Image Intensity Range:", sitk.GetArrayFromImage(fixed_image).min(), 
          sitk.GetArrayFromImage(fixed_image).max())
    print("Moving Image Intensity Range:", sitk.GetArrayFromImage(moving_image).min(), 
          sitk.GetArrayFromImage(moving_image).max())

    # Ensure moving image has the same size as the fixed image
    new_size = fixed_image.GetSize()
    moving_image = rescale_image(moving_image, new_size)

    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)
    demons_filter.SetStandardDeviations(demons_std)

    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: print('\r{0}'.format(demons_filter.GetElapsedIterations()), end=''))
    
    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    resampled_moving_image = resample_moving_image(fixed_image, moving_image, demons_transform)

    # Construct file paths
    demons_registration_path = os.path.join(output_path, f"demons_registration_{mask_name}.mha")
    demons_displacement_path = os.path.join(output_path, f"demons_displacement_field_{mask_name}.mha")
    
    # Write output
    sitk.WriteImage(resampled_moving_image, demons_registration_path)
    sitk.WriteImage(demons_transform, demons_displacement_path)
    
    return demons_transform, resampled_moving_image

def resample_moving_image(fixed_image, moving_image, displacement_field):
    """
    Resample the moving image using the displacement field.
    
    Parameters:
    - fixed_image: Target image for resampling.
    - moving_image: Image to be resampled.
    - displacement_field: Displacement field from Demons registration.
    
    Returns:
    - Resampled moving image.
    """
    resampler = sitk.WarpImageFilter()
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(moving_image, displacement_field)
