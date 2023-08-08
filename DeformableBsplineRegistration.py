# In[1]:
import os
import SimpleITK as sitk
from utils import *


# In[2]:
# DeformableBsplineRegistration.py

def normalize_image(img):
    """Normalize the image intensities using the provided formula.
    
    Args:
        img (SimpleITK.Image): The input image.
    
    Returns:
        SimpleITK.Image: The normalized image.
    """
    # Convert the SimpleITK Image to a numpy array
    img_array = sitk.GetArrayViewFromImage(img)
    
    # Normalize the array based on the provided formula
    normalized_img_array = (img_array - img_array.mean()) / img_array.std()
    
    # Convert the normalized numpy array back to a SimpleITK Image
    return sitk.GetImageFromArray(normalized_img_array)

def setup_deformable_registration(fixed_image):
    """Initialize and set up the deformable registration method.
    
    Args:
        fixed_image (SimpleITK.Image): The fixed (reference) image.
    
    Returns:
        SimpleITK.ImageRegistrationMethod: Configured registration method.
    """
    deformable_registration_method = sitk.ImageRegistrationMethod()

    # Use the correlation as the metric for optimization
    deformable_registration_method.SetMetricAsCorrelation()



    # Use the LBFGSB optimization algorithm
    deformable_registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)

    # Use linear interpolation for the images during the optimization process
    deformable_registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set up a multi-resolution strategy to start with a coarse resolution and refine in subsequent stages
    deformable_registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    deformable_registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    deformable_registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return deformable_registration_method

def perform_deformable_bspline_registration(fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image):
    """Perform deformable B-spline registration.
    
    Args:
        fixed_image (SimpleITK.Image): The fixed (reference) image.
        moving_image (SimpleITK.Image): The image to be registered to the fixed image.
        output_path (str): Path to save the registration results.
        final_transform_v1 (SimpleITK.Transform): Initial transformation (likely from previous registration steps).
        resampled_moving_image (SimpleITK.Image): Moving image after initial registration.

    Returns:
        tuple: Tuple containing the final deformable transform and the resampled moving image.
    """

    # Normalize the fixed and moving images before registration
    fixed_image = normalize_image(fixed_image)
    moving_image = normalize_image(moving_image)

    # Convert the SimpleITK Image to a numpy array
    fixed_image_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_image_array = sitk.GetArrayViewFromImage(moving_image)

    # Display the min and max pixel values for the fixed and moving images
    print("Fixed Image Pixel Values - Min:", fixed_image_array.min(), "Max:", fixed_image_array.max())
    print("Moving Image Pixel Values - Min:", moving_image_array.min(), "Max:", moving_image_array.max())

    # Check if previous results exist to avoid redundant computation
    if os.path.exists(output_path + "/bspline_deformable_transformation.tfm") and \
       os.path.exists(output_path + "/composite_transform.tfm") and \
       os.path.exists(output_path + "/bspline_deformable_registration.mha"):
        
        # Read previous results from disk
        final_deformable_transform = sitk.ReadTransform(output_path + "/bspline_deformable_transformation.mha")
        composite_transform = sitk.ReadTransform(output_path + "/composite_transform.mha")
        resampled_moving_image_deformable = sitk.ReadImage(output_path + "/bspline_deformable_registration.mha")
    else:
        # Setup the deformable registration method
        deformable_registration_method = setup_deformable_registration(fixed_image)

        # Initialize the B-spline transform with given parameters
        transform_domain_physical_dim_size = fixed_image.GetSize()
        transform_domain_mesh_size = [size // 4 for size in transform_domain_physical_dim_size]
        initial_deformable_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                                        transformDomainMeshSize=transform_domain_mesh_size, order=3)

        # Set the initial transform for the registration process
        deformable_registration_method.SetInitialTransform(initial_deformable_transform)
        
        # Add a command to be called at every iteration for logging or updating UI
        deformable_registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(deformable_registration_method))

        # Execute the registration method
        final_deformable_transform = deformable_registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                                            sitk.Cast(resampled_moving_image, sitk.sitkFloat32))

        # Combine the initial transformation and the computed deformable transformation
        composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
        composite_transform.AddTransform(final_transform_v1)
        composite_transform.AddTransform(final_deformable_transform)

        # Save the transform to disk
        sitk.WriteTransform(final_deformable_transform, output_path + "/bspline_deformable_transformation.tfm")

        # Resample the moving image using the composite transformation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetTransform(composite_transform)
        resampled_moving_image_deformable = resampler.Execute(resampled_moving_image)

        # Save the resampled moving image to disk
        sitk.WriteImage(resampled_moving_image_deformable, output_path + "/bspline_deformable_registration.mha")

    # Display the min and max pixel values for the resampled moving image
    print("Resampled Moving Image Pixel Values - Min:", sitk.GetArrayViewFromImage(resampled_moving_image_deformable).min(), "Max:", sitk.GetArrayViewFromImage(resampled_moving_image_deformable).max())

    return final_deformable_transform, resampled_moving_image_deformable
