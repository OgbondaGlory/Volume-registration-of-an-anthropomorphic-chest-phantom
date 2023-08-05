# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DemonsRegistration.py

def apply_demons_algorithm(fixed_image, moving_image, output_path, iterations=100):
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)

    print("********************************************************************************")
    print("* Demons registration                                                           ")
    print("********************************************************************************")
    # You may need to adjust the callback function based on your specific implementation
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    
    # Resample the moving image using the demons transform
    resampled_moving_image = resample_moving_image(fixed_image, moving_image, demons_transform)
    
    # Write out the resampled moving image and the demons transform
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(output_path, "demons_registration.mha"))
    writer.Execute(resampled_moving_image)
    
    sitk.WriteTransform(demons_transform, os.path.join(output_path, "demons_transformation.tfm"))
    
    return demons_transform, resampled_moving_image


def resample_moving_image(fixed_image, moving_image, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    return resampler.Execute(moving_image)
