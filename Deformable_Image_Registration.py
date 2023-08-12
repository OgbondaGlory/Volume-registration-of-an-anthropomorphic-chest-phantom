#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import SimpleITK as sitk
from voxelmorph.torch.networks import VxmDense

from utils import *
from RigidDeformation import perform_rigid_registration
from DeformableBsplineRegistration import perform_deformable_bspline_registration
from DemonsRegistration import apply_demons_algorithm, resample_moving_image
from DNNRegistration import apply_dnn_registration
from TrainingDNN import train_dnn_model

def Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path, operation):

    # print(0)
    # Load the DICOM images for both patient and phantom
    patient_ct_image = load_dicom_series(patient_image_path)
    phantom_ct_image = load_dicom_series(phantom_image_path)

    # Paths for segmentation masks
    patient_lung_mask_path = os.path.join(output_path, os.path.basename(patient_image_path) + "_lung_mask.mha")
    phantom_lung_mask_path = os.path.join(output_path, "Phantom_CT_Scan_lung_mask.mha")
    
    patient_bone_mask_path = os.path.join(output_path, os.path.basename(patient_image_path) + "_bone_mask.mha")
    phantom_bone_mask_path = os.path.join(output_path, "Phantom_CT_Scan_bone_mask.mha")

    # Load or generate lung segmentation masks
    if os.path.exists(patient_lung_mask_path) and os.path.exists(phantom_lung_mask_path):
        # Load masks from disk
        patient_lung_mask = sitk.ReadImage(patient_lung_mask_path)
        phantom_lung_mask = sitk.ReadImage(phantom_lung_mask_path)
        
        # Load bone masks from disk
        patient_bone_mask = sitk.ReadImage(patient_bone_mask_path)
        phantom_bone_mask = sitk.ReadImage(phantom_bone_mask_path)
    else:
        if operation == 'segment':
            # Segment lung on both patient and phantom CT images
            patient_lung_mask = segment_lung(patient_ct_image)
            phantom_lung_mask = segment_lung(phantom_ct_image)

            # Save lung segmentation results
            save_segmentation(patient_lung_mask, patient_image_path, output_path, "lung")
            save_segmentation(phantom_lung_mask, phantom_image_path, output_path, "lung")

            # Perform bone segmentation
            patient_bone_mask = segment_bones(patient_ct_image)
            phantom_bone_mask = segment_bones(phantom_ct_image)

            # Save bone segmentation results
            save_segmentation(patient_bone_mask, patient_image_path, output_path, "bone")
            save_segmentation(phantom_bone_mask, phantom_image_path, output_path, "bone")
    # At this point, you have patient_lung_mask and phantom_lung_mask ready for use, whether they were loaded or segmented just now.
    if operation != 'segment':
    # Define masks for the two registration pipelines
    # For the demons registration, we will use the original grayscale images.
    # For other operations, we'll continue to use the masks.
        if operation == 'demons':
            fixed_images = [patient_ct_image, patient_ct_image]
            moving_images = [phantom_ct_image, phantom_ct_image]
        else:
            fixed_images = [patient_lung_mask, patient_bone_mask]
            moving_images = [phantom_lung_mask, phantom_bone_mask]

        mask_names = ['lung', 'bone']
        for idx, (fixed_image, moving_image, mask_name) in enumerate(zip(fixed_images, moving_images, mask_names)):
        # writer = sitk.ImageFileWriter()

            if operation == 'rigid':
                print("Performing rigid registration...")
                final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path, mask_name)
                print("Rigid registration completed.")
                # Generate checkerboard for Rigid Registration
                checker_image = generate_checkerboard(fixed_image, resampled_moving_image)
                checker_name = f"checkerboard_rigid_registration_{mask_name}"
                save_images(checker_image, output_path, checker_name)
                # Display the checkerboard image for Rigid Registration
                display_images(checker_image, "Checkerboard for Rigid Registration")

                # For ISO Surfaces for Rigid Registration
                output_resampled_image_path = generate_filename(output_path, 'rigid', mask_name)
                output_iso_surface_file_path = generate_filename(output_path, f"iso_surface_rigid_{mask_name}", "stl")
                resampled_moving_image = sitk.ReadImage(output_resampled_image_path)
                verts, faces = extract_iso_surface(resampled_moving_image, level=0.5, smooth=0.0)
                save_iso_surface(verts, faces, output_iso_surface_file_path)

            elif operation == 'bspline':
                  print("Performing deformable B-spline registration...")
                  # Check if final_transform_v1 and resampled_moving_image from rigid operation are available
                  rigid_transform_file = os.path.join(output_path, "rigid_transformation.tfm")
                  resampled_moving_image_file = os.path.join(output_path, "rigid_registration.mha")
                  if os.path.exists(rigid_transform_file) and os.path.exists(resampled_moving_image_file):
                      print("Loading results from previous rigid registration...")
                      final_transform_v1 = sitk.ReadTransform(rigid_transform_file)
                      resampled_moving_image = sitk.ReadImage(resampled_moving_image_file)
                  else:
                      print("Performing preliminary rigid registration...")
                      final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path, mask_name)
                      print("Preliminary rigid registration completed.")
                        
                  resampled_moving_image_deformable = perform_deformable_bspline_registration(
                  fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image, mask_name)
                  print("Deformable B-spline registration completed.")
                  # Generate checkerboard for B-Spline deformation
                  checker_image_deformable = generate_checkerboard(fixed_image, resampled_moving_image_deformable)
                  checker_name = f"checkerboard_bspline_deformation_{mask_name}"
                  save_images(checker_image_deformable, output_path, checker_name)                # Display the checkerboard image for B-Spline deformation
                  display_images(checker_image_deformable, "Checkerboard for B-Spline deformation")

                  # Extracting the ISO Surfaces for B-spline
                  output_resampled_image_path = generate_filename(output_path, 'bspline_deformable', mask_name)
                  output_iso_surface_file_path = generate_filename(output_path, f"iso_surface_deformable_{mask_name}", "stl")
                  resampled_moving_image_deformable = sitk.ReadImage(output_resampled_image_path)
                  verts, faces = extract_iso_surface(resampled_moving_image_deformable, level=0.5, smooth=0.0)
                  save_iso_surface(verts, faces, output_iso_surface_file_path)

            elif operation == 'demons':
                  print("Applying Demons algorithm...")
                    
                  output_resampled_image_path = os.path.join(output_path, "demons_registration.mha")
                  output_demons_transform_path = os.path.join(output_path, "demons_transformation.tfm")
                    
                  if os.path.exists(output_resampled_image_path) and os.path.exists(output_demons_transform_path):
                     print("Found existing Demons transformation, reading from disk.")
                     resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)
                     demons_transform = sitk.ReadTransform(output_demons_transform_path)
                  else:
                      # Applying the Demons algorithm and saving the output
                      demons_transform, resampled_moving_image_demons = apply_demons_algorithm(fixed_image, moving_image, output_path, mask_name)
                      print("Demons algorithm completed.")
                    
                      # Display the images after transformation
                      display_images(fixed_image, "Fixed Image after Demons Transformation")
                      display_images(resampled_moving_image_demons, "Resampled Moving Image after Demons Transformation")

                      # Generate checkerboard for Demons registration
                      checker_image_demons = generate_checkerboard(fixed_image, resampled_moving_image_demons)
                      checker_name = f"checkerboard_demons_registration_{mask_name}"
                      save_images(checker_image_demons, output_path, checker_name)                    
                      # Display the checkerboard image for Demons registration
                      display_images(checker_image_demons, "Checkerboard for Demons registration")

                      #
                      # Extracting the ISO Surfaces for Demons
                      output_resampled_image_path = generate_filename(output_path, 'demons', mask_name)
                      output_iso_surface_file_path = generate_filename(output_path, f"iso_surface_demons_{mask_name}", "stl")
                      resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)
                      verts, faces = extract_iso_surface(resampled_moving_image_demons, level=0.5, smooth=0.0)
                      save_iso_surface(verts, faces, output_iso_surface_file_path)

            elif operation == 'dnn':
                  # Applying CNNS
                  # Use the paths passed to the function instead of hardcoding them
                  phantom_directory_path = phantom_image_path
                  patient_directory_path = patient_image_path
                    
                  print("Applying CNNS registration...")
                  transformed_moving_image = apply_dnn_registration(output_path, phantom_directory_path, patient_directory_path, mask_name)
                  print("CNNS registration completed.")
                  # Generate checkerboard for CNN registration
                  checker_image_cnn = generate_checkerboard(fixed_image, transformed_moving_image)
                  checker_name = f"checkerboard_cnn_registration_{mask_name}"
                  save_images(checker_image_cnn, output_path, checker_name)                # Display the checkerboard image for CNN registration
                  display_images(checker_image_cnn, "Checkerboard for CNN registration")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise IOError("Invalid cmd line,\nUsage: " + sys.argv[0] + "   DICOM_PATH   OUTPUT_PATH   OPERATION")

    patient_image_path = sys.argv[1]
    phantom_image_path = r"Phantom_CT_Scan"
    output_path = sys.argv[2]
    operation = sys.argv[3]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path, operation)

