#!/usr/bin/env python3
# coding: utf-8

# # 3D-3D Medical Imaging Segmentation, Rigid, and Non Registration of the Human Chest Using Classing Techniques and DNN 

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

# !pip install numpy-stl

def Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path):
    # Load the DICOM images
    ct_image = load_dicom_series(phantom_image_path)

    # Perform lung segmentation
    lung_mask = segment_lung(ct_image)

    # Perform bone segmentation
    bone_mask = segment_bones(ct_image)

    # Save segmented lung and bone images
    sitk.WriteImage(lung_mask, os.path.join(output_path, "lung_segmentation.mhd"))
    sitk.WriteImage(bone_mask, os.path.join(output_path, "bone_segmentation.mhd"))

    # Load the DICOM images
    fixed_image = load_dicom_series(patient_image_path)
    moving_image = load_dicom_series(phantom_image_path)

    # reader = sitk.ImageFileReader()
    writer = sitk.ImageFileWriter()

    if not os.path.exists(output_path + "/target.mha"):
        writer.SetFileName(output_path + "/target.mha")
        writer.Execute(fixed_image)

    if not os.path.exists(output_path + "/source.mha"):
        writer.SetFileName(output_path + "/source.mha")
        writer.Execute(moving_image)

    # Display the images with appropriate titles
    display_images(fixed_image, "Fixed Image patient's CT scan")
    display_images(moving_image, "Moving Image CT scan of the phantom")

    print("Performing rigid registration...")
    final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path)
    print("Rigid registration completed.")

    # Display the images after transformation
    display_images(fixed_image, "Fixed Image after Transformation")
    display_images(resampled_moving_image, "Resampled Moving Image after Transformation")
    # Generate checkerboard for Rigid Registration
    checker_image = generate_checkerboard(fixed_image, resampled_moving_image)
    # Display the checkerboard image for Rigid Registration
    display_images(checker_image, "Checkerboard for Rigid Registration")
    
    # Extracting the ISO Surfaces for Rigid Registration
    output_resampled_image_path = os.path.join(output_path, "rigid_registration.mha")
    output_iso_surface_file_path = os.path.join(output_path, "iso_surface_rigid.stl")
    resampled_moving_image = sitk.ReadImage(output_resampled_image_path)
    verts, faces = extract_iso_surface(resampled_moving_image, level=0.5, smooth=0.0)
    save_iso_surface(verts, faces, output_iso_surface_file_path)

    # Deformable Registration
    # Call the deformable B-spline registration function
    print("Performing deformable B-spline registration...")
    resampled_moving_image_deformable = perform_deformable_bspline_registration(
        fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image)
    print("Deformable B-spline registration completed.")
    
    # Display the images after transformation
    display_images(fixed_image, "Fixed Image after Transformation")
    display_images(resampled_moving_image, "Resampled Moving Image after Transformation")
    # Generate checkerboard for B-Spline deformation
    checker_image_deformable = generate_checkerboard(fixed_image, resampled_moving_image_deformable)
    # Display the checkerboard image for B-Spline deformation
    display_images(checker_image_deformable, "Checkerboard for B-Spline deformation")

    # Extracting the ISO Surfaces for B-spline
    output_resampled_image_path = os.path.join(output_path, "deformable_registration.mha")
    output_iso_surface_file_path = os.path.join(output_path, "iso_surface_deformable.stl")
    resampled_moving_image_deformable = sitk.ReadImage(output_resampled_image_path)
    verts, faces = extract_iso_surface(resampled_moving_image_deformable, level=0.5, smooth=0.0)
    save_iso_surface(verts, faces, output_iso_surface_file_path)

   # Apply Demons Algorithm
    print("Applying Demons algorithm...")
    demons_transform = apply_demons_algorithm(fixed_image, resampled_moving_image)
    print("Demons algorithm completed.")
    
    output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")
    if os.path.exists(output_resampled_image_path):
        resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)
    else:
        resampled_moving_image_demons = resample_moving_image(fixed_image, moving_image, demons_transform)
        writer.SetFileName(output_resampled_image_path)
        writer.Execute(resampled_moving_image_demons)

    # Display the images after transformation
    display_images(fixed_image, "Fixed Image after Demons Transformation")
    display_images(resampled_moving_image_demons, "Resampled Moving Image after Demons Transformation")
    # Generate checkerboard for Demons registration
    checker_image_demons = generate_checkerboard(fixed_image, resampled_moving_image_demons)
    # Display the checkerboard image for Demons registration
    display_images(checker_image_demons, "Checkerboard for Demons registration")

    # Extracting the ISO Surfaces for Demons
    output_resampled_image_path = os.path.join(output_path, "resampled_moving_image_demons.mha")
    output_iso_surface_file_path = os.path.join(output_path, "iso_surface_demons.stl")
    resampled_moving_image_demons = sitk.ReadImage(output_resampled_image_path)
    verts, faces = extract_iso_surface(resampled_moving_image_demons, level=0.5, smooth=0.0)
    save_iso_surface(verts, faces, output_iso_surface_file_path)

    # Applying CNNS
    phantom_directory_path = r"Phantom_CT_Scan"
    patient_directory_path = r"Patient_CT_Scan"
    
     # Call the function to apply CNNS and get the transformed moving image
    print("Applying CNNS registration...")
    transformed_moving_image = apply_dnn_registration(output_path, phantom_directory_path, patient_directory_path)
    print("CNNS registration completed.")
    
    # Generate checkerboard for CNN registration
    checker_image_cnn = generate_checkerboard(fixed_image, transformed_moving_image)
    # Display the checkerboard image for CNN registration
    display_images(checker_image_cnn, "Checkerboard for CNN registration")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise IOError("Invalid cmd line,\nUsage: " + sys.argv[0] + "   DICOM_PATH   OUTPUT_PATH")

    patient_image_path = sys.argv[1]
    phantom_image_path = r"Phantom_CT_Scan"
    output_path = sys.argv[2]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path)
