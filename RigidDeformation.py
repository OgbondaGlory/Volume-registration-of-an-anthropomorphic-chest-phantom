# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# RigidDeformation.py

def perform_rigid_registration(fixed_image, moving_image, output_path):
    if os.path.exists(output_path + "/rigid_transformation.tfm") and os.path.exists(output_path + "/rigid_registration.mha"):
        final_transform_v1 = sitk.ReadTransform(output_path + "/rigid_transformation.tfm")
        reader = sitk.ImageFileReader()
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

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path + "/rigid_registration.mha")
        writer.Execute(resampled_moving_image)

    return final_transform_v1, resampled_moving_image
