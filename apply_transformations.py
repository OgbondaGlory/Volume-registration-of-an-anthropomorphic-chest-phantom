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
    return label_to_hu

def read_transform_parameters(transform_path):
    try:
        with open(transform_path, 'r') as tfm_file:
            lines = tfm_file.readlines()
        for line in lines:
            if 'Parameters:' in line:
                parameters = line.split(':')[1].strip().split(' ')
                return [float(p) for p in parameters]
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

def map_to_hu_values(label_image, label_to_hu):
    label_array = sitk.GetArrayFromImage(label_image)
    hu_mapped_array = np.copy(label_array)
    for label, hu_value in label_to_hu.items():
        hu_mapped_array[label_array == label] = hu_value
    hu_mapped_image = sitk.GetImageFromArray(hu_mapped_array)
    hu_mapped_image.CopyInformation(label_image)
    return hu_mapped_image

def load_patient_ct_scan(directory_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
    if not dicom_names:
        print(f"No DICOM series found in directory: {directory_path}")
        return None
    reader.SetFileNames(dicom_names)
    try:
        return reader.Execute()
    except RuntimeError as e:
        print(f"Error reading DICOM series from {directory_path}: {e}")
        return None

def apply_transformations(patient_directory, labels_directory, dat_file_path, label_filename="labels.mha"):
    patient_ct_image = load_patient_ct_scan(patient_directory)
    if patient_ct_image is None:
        return

    label_path = os.path.join(labels_directory, label_filename)
    label_image = sitk.ReadImage(label_path)
    label_to_hu = parse_dat_file(dat_file_path)
    hu_mapped_label_image = map_to_hu_values(label_image, label_to_hu)

    transform_path = os.path.join(patient_directory, "rigid_transformation.tfm")
    parameters = read_transform_parameters(transform_path)
    if parameters:
        apply_rigid_body_transformation(hu_mapped_label_image, parameters, "rigid_body_labels", patient_directory)
        apply_rigid_body_transformation(patient_ct_image, parameters, "rigid_body_patient_ct", patient_directory)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations.py [patient_directories] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directories = sys.argv[1].split(',')  # Expecting a comma-separated list of patient directories
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]

    for patient_directory in patient_directories:
        apply_transformations(patient_directory, labels_directory, dat_file_path)
