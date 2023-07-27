# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DemonsRegistration.py

def apply_demons_algorithm(fixed_image, moving_image, iterations=100):
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(iterations)

    print("********************************************************************************")
    print("* Demons registration                                                           ")
    print("********************************************************************************")
    # You may need to adjust the callback function based on your specific implementation
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

    demons_transform = demons_filter.Execute(fixed_image, moving_image)
    return demons_transform

def resample_moving_image(fixed_image, moving_image, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    return resampler.Execute(moving_image)
