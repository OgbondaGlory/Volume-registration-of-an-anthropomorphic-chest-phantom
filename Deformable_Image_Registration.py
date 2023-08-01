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

def Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path, operations):
      # Load the DICOM images for both patient and phantom
    patient_ct_image = load_dicom_series(patient_image_path)
    phantom_ct_image = load_dicom_series(phantom_image_path)

    # Perform lung segmentation
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

    # Load the DICOM images
    fixed_image = load_dicom_series(patient_image_path)
    moving_image = load_dicom_series(phantom_image_path)

    writer = sitk.ImageFileWriter()

    if operation == 'rigid':
        print("Performing rigid registration...")
        final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path)
        print("Rigid registration completed.")
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

    if operation == 'bspline':
        print("Performing deformable B-spline registration...")
        resampled_moving_image_deformable = perform_deformable_bspline_registration(
            fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image)
        print("Deformable B-spline registration completed.")
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

    if operation == 'demons':
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

    if operation == 'dnn':
        # Applying CNNS
        phantom_directory_path = r"Phantom_CT_Scan"
        patient_directory_path = r"Patient_CT_Scan"
        
        print("Applying CNNS registration...")
        transformed_moving_image = apply_dnn_registration(output_path, phantom_directory_path, patient_directory_path)
        print("CNNS registration completed.")
        # Generate checkerboard for CNN registration
        checker_image_cnn = generate_checkerboard(fixed_image, transformed_moving_image)
        # Display the checkerboard image for CNN registration
        display_images(checker_image_cnn, "Checkerboard for CNN registration")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise IOError("Invalid cmd line,\nUsage: " + sys.argv[0] + "   DICOM_PATH   OUTPUT_PATH   OPERATION")

    patient_image_path = sys.argv[1]
    phantom_image_path = r"Phantom_CT_Scan"
    output_path = sys.argv[2]
    operations = sys.argv[3]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for operation in operations:
        Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path, operation)