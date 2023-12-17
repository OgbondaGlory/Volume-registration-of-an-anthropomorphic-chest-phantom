#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import SimpleITK as sitk
from utils import *
from RigidDeformation import perform_rigid_registration
from DeformableBsplineRegistration import perform_deformable_bspline_registration
from DemonsRegistration import apply_demons_algorithm
from DNNRegistration import apply_dnn_registration

def Deformable_Image_Registration(patient_image_path, phantom_image_path, output_path, operation):
    # Load DICOM images
    patient_ct_image = load_dicom_series(patient_image_path)
    phantom_ct_image = load_dicom_series(phantom_image_path)

    fixed_image = patient_ct_image
    moving_image = phantom_ct_image

    if operation == 'rigid':
        print("Performing rigid registration...")
        final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path, "full_image")

        # Generate and save checkerboard and ISO surfaces
        checker_image = generate_checkerboard(fixed_image, resampled_moving_image)
        save_images(checker_image, output_path, "checkerboard_rigid_full_image")
        verts, faces = extract_iso_surface(resampled_moving_image, level=0.5)
        save_iso_surface(verts, faces, os.path.join(output_path, "iso_surface_rigid_full_image.stl"))

    elif operation == 'bspline':
        print("Performing deformable B-spline registration...")
        final_transform_v1, resampled_moving_image = perform_rigid_registration(fixed_image, moving_image, output_path, "preliminary")
        resampled_moving_image_deformable = perform_deformable_bspline_registration(fixed_image, moving_image, output_path, final_transform_v1, resampled_moving_image, "full_image")

        checker_image = generate_checkerboard(fixed_image, resampled_moving_image_deformable)
        save_images(checker_image, output_path, "checkerboard_bspline_full_image")
        verts, faces = extract_iso_surface(resampled_moving_image_deformable, level=0.5)
        save_iso_surface(verts, faces, os.path.join(output_path, "iso_surface_bspline_full_image.stl"))

    elif operation == 'demons':
        print("Applying Demons algorithm...")
        demons_transform, resampled_moving_image_demons = apply_demons_algorithm(fixed_image, moving_image, output_path, "full_image")

        checker_image = generate_checkerboard(fixed_image, resampled_moving_image_demons)
        save_images(checker_image, output_path, "checkerboard_demons_full_image")
        verts, faces = extract_iso_surface(resampled_moving_image_demons, level=0.5)
        save_iso_surface(verts, faces, os.path.join(output_path, "iso_surface_demons_full_image.stl"))

    elif operation == 'dnn':
        print("Applying CNNS registration...")
        transformed_moving_image = apply_dnn_registration(output_path, phantom_image_path, patient_image_path, "full_image")

        checker_image = generate_checkerboard(fixed_image, transformed_moving_image)
        save_images(checker_image, output_path, "checkerboard_cnn_full_image")
        verts, faces = extract_iso_surface(transformed_moving_image, level=0.5)
        save_iso_surface(verts, faces, os.path.join(output_path, "iso_surface_cnn_full_image.stl"))

    else:
        raise ValueError("Invalid operation specified.")

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
