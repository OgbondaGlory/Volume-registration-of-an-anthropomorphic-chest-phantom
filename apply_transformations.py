import SimpleITK as sitk
import numpy as np
import os
import sys

def parse_dat_file(dat_file_path):
    label_to_hu = {}
    with open(dat_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label = int(parts[0])
                hu_value = float(parts[1])
                label_to_hu[label] = hu_value
                print(f"Mapping label {label} to HU value {hu_value}")
    return label_to_hu

def read_transform_parameters(transform_path):
    try:
        with open(transform_path, 'r') as tfm_file:
            lines = tfm_file.readlines()
        for line in lines:
            if 'Parameters:' in line:
                parameters = line.split(':')[1].strip().split(' ')
                parameters = [float(p) for p in parameters]
                return parameters
    except FileNotFoundError:
        print(f"Transformation file not found: {transform_path}")
    return None

def apply_rigid_body_transformation(image, parameters, transform_name, output_path):
    transform = sitk.Euler3DTransform()
    transform.SetParameters(parameters[:6])
    transformed_image = sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor, 0.0, image.GetPixelID())
    save_transformed_image(transformed_image, transform_name, output_path)

def save_transformed_image(transformed_image, transform_name, output_path):
    transformed_image_path = os.path.join(output_path, f"{transform_name}_transformed.mha")
    sitk.WriteImage(transformed_image, transformed_image_path)
    print(f"Transformed image saved at {transformed_image_path}")

def map_to_hu_values(label_image, label_to_hu, output_path):
    label_array = sitk.GetArrayFromImage(label_image)
    hu_mapped_array = np.copy(label_array)
    for label, hu_value in label_to_hu.items():
        hu_mapped_array[label_array == label] = hu_value
    hu_mapped_image = sitk.GetImageFromArray(hu_mapped_array)
    hu_mapped_image.CopyInformation(label_image)
    hu_mapped_image_path = os.path.join(output_path, "hu_mapped_labels.mha")
    sitk.WriteImage(hu_mapped_image, hu_mapped_image_path)
    print(f"HU-mapped label image saved at {hu_mapped_image_path}")

def apply_transformations(patient_directory, labels_directory, dat_file_path, label_filename="labels.mha"):
    # Apply transformation to labels
    label_path = os.path.join(labels_directory, label_filename)
    label_image = sitk.ReadImage(label_path)
    label_to_hu = parse_dat_file(dat_file_path)
    map_to_hu_values(label_image, label_to_hu, patient_directory)

    # Apply transformation to patient CT scan
    patient_ct_path = os.path.join(patient_directory, "patient_ct.mha")  # Replace with actual patient CT file name
    patient_ct_image = sitk.ReadImage(patient_ct_path)

    # Read transform parameters and apply rigid body transformation
    transform_path = os.path.join(patient_directory, "rigid_transformation.tfm")
    parameters = read_transform_parameters(transform_path)
    if parameters:
        apply_rigid_body_transformation(label_image, parameters, "rigid_body_labels", patient_directory)
        apply_rigid_body_transformation(patient_ct_image, parameters, "rigid_body_patient_ct", patient_directory)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations.py [patient_directory] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directory = sys.argv[1]
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]
    apply_transformations(patient_directory, labels_directory, dat_file_path)
