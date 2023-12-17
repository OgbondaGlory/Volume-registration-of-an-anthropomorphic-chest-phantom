# In[1]:
#!/usr/bin/env python3
# coding: utf-8

import os
import SimpleITK as sitk
from utils import *

def perform_rigid_registration(fixed_image, moving_image, output_path):
    print("Fixed Image Intensity Range:", sitk.GetArrayFromImage(fixed_image).min(), sitk.GetArrayFromImage(fixed_image).max())
    print("Moving Image Intensity Range:", sitk.GetArrayFromImage(moving_image).min(), sitk.GetArrayFromImage(moving_image).max())

    try:
        # Construct the file paths for saving and loading the results
        rigid_transform_path = os.path.join(output_path, "rigid_transformation.tfm")
        rigid_registration_image_path = os.path.join(output_path, "rigid_registration.mha")

        # Check if results already exist on disk
        if os.path.exists(rigid_transform_path) and os.path.exists(rigid_registration_image_path):
            # Load the transformation and the registered image from the disk
            final_transform_v1 = sitk.ReadTransform(rigid_transform_path)
            reader = sitk.ImageFileReader()
            reader.SetFileName(rigid_registration_image_path)
            resampled_moving_image = reader.Execute()

        else:
            # Perform registration if results do not exist
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsCorrelation()
            registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Initialize transform
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.VersorRigid3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            print("********************************************************************************")
            print("* Rigid registration                                                            ")
            print("********************************************************************************")
            
            # Execute registration
            final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

            # Save transformation and resampled image
            sitk.WriteTransform(final_transform_v1, rigid_transform_path)
            resampled_moving_image = resample_moving_image(moving_image, final_transform_v1, fixed_image)
            sitk.WriteImage(resampled_moving_image, rigid_registration_image_path)

        print("Rigid registration completed.")
        return final_transform_v1, resampled_moving_image
    except Exception as e:
        print("Error in rigid registration:", str(e))
        return None, None

def resample_moving_image(moving_image, transform, fixed_image):
    # Resample the moving image to align with the fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    return resampler.Execute(moving_image)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise IOError("Invalid cmd line,\nUsage: " + sys.argv[0] + " FIXED_IMAGE_PATH MOVING_IMAGE_PATH OUTPUT_PATH")

    fixed_image_path = sys.argv[1]
    moving_image_path = sys.argv[2]
    output_path = sys.argv[3]

    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    final_transform, resampled_image = perform_rigid_registration(fixed_image, moving_image, output_path)
    # Additional code to process or display final_transform and resampled_image can be added here
