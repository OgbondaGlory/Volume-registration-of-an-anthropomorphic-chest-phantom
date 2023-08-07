# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DemonsRegistration.py

# DemonsRegistration.py

# Function to rescale the image
def rescale_image(image, new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(
        [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in zip(image.GetSize(), image.GetSpacing(), new_size)])
    resampler.SetOutputDirection(image.GetDirection())
    return resampler.Execute(image)

def iteration_callback(filter):
    print('\r{0}'.format(filter.GetElapsedIterations()), end='')

def apply_demons_algorithm(fixed_image, moving_image, output_path, iterations=100):
    # Ensure that the moving image has the same size as the fixed image
    new_size = fixed_image.GetSize()
    moving_image = rescale_image(moving_image, new_size)

    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)

    print("********************************************************************************")
    print("* Demons registration                                                           ")
    print("********************************************************************************")
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    
    # Resample the moving image using the demons transform
    resampled_moving_image = resample_moving_image(fixed_image, moving_image, demons_transform)
    
    # Write out the resampled moving image and the displacement field
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(output_path, "demons_registration.mha"))
    writer.Execute(resampled_moving_image)
    print(type(demons_transform))
    
    # Save the displacement field as an image
    sitk.WriteImage(demons_transform, os.path.join(output_path, "demons_displacement_field.mha"))
    
    return demons_transform, resampled_moving_image


def resample_moving_image(fixed_image, moving_image, displacement_field):
    print("Debug info: ", type(fixed_image), fixed_image.GetSize())  # Debug line
    resampler = sitk.WarpImageFilter()
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_moving_image = resampler.Execute(moving_image, displacement_field)
    return resampled_moving_image
