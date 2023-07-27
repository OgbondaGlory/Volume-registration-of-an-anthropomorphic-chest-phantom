# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DeformableBsplineRegistration.py


def perform_deformable_bspline_registration(fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image):
    if os.path.exists(output_path + "/deformable_transformation.tfm") and os.path.exists(output_path + "/composite_transform.tfm") and os.path.exists(output_path + "/deformable_registration.mha"):
        final_deformable_transform = sitk.ReadTransform(output_path + "/deformable_transformation.mha")
        composite_transform = sitk.ReadTransform(output_path + "/composite_transform.mha")

        reader = sitk.ImageFileReader()
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
        deformable_registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        deformable_registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        deformable_registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        deformable_registration_method.SetInitialTransform(initial_deformable_transform)

        print("********************************************************************************")
        print("* Deformable registration                                                       ")
        print("********************************************************************************")
        deformable_registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(deformable_registration_method))

        final_deformable_transform = deformable_registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                                            sitk.Cast(resampled_moving_image, sitk.sitkFloat32))

        # Combine the affine and deformable transforms
        composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
        composite_transform.AddTransform(final_transform_v1)
        composite_transform.AddTransform(final_deformable_transform)

        sitk.WriteTransform(final_deformable_transform, output_path + "/deformable_transformation.tfm")

        # Resample the moving image onto the fixed image's grid using the composite transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetTransform(composite_transform)
        resampled_moving_image_deformable = resampler.Execute(resampled_moving_image)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path + "/deformable_registration.mha")
        writer.Execute(resampled_moving_image_deformable)

    return final_deformable_transform, resampled_moving_image_deformable


