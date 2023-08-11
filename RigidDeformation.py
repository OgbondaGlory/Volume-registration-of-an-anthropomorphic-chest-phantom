# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# RigidDeformation.py

def perform_rigid_registration(fixed_image, moving_image, output_path, mask_name):
    # Construct the file paths for saving and loading the results
    rigid_transform_path = os.path.join(output_path, f"rigid_transformation_{mask_name}.tfm")
    rigid_registration_image_path = os.path.join(output_path, f"rigid_registration_{mask_name}.mha")

    # Check if results already exist on disk
    if os.path.exists(rigid_transform_path) and os.path.exists(rigid_registration_image_path):
        # Load the transformation and the registered image from the disk
        final_transform_v1 = sitk.ReadTransform(rigid_transform_path)
        reader = sitk.ImageFileReader()
        reader.SetFileName(rigid_registration_image_path)
        resampled_moving_image = reader.Execute()
    else:
        # If results do not exist, perform the registration
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Initialization
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        print("********************************************************************************")
        print("* Rigid registration                                                            ")
        print("********************************************************************************")
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

        # Execute registration
        final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

        # Save the transformation to disk
        sitk.WriteTransform(final_transform_v1, rigid_transform_path)

        # Resample the moving image onto the fixed image's grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetTransform(final_transform_v1)
        resampled_moving_image = resampler.Execute(moving_image)

        # Save the resampled image to disk
        writer = sitk.ImageFileWriter()
        writer.SetFileName(rigid_registration_image_path)
        writer.Execute(resampled_moving_image)

    # Return the transformation and the resampled image
    return final_transform_v1, resampled_moving_image

