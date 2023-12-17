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

def load_patient_ct_scan(directory_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
    reader.SetFileNames(dicom_names)
    try:
        image = reader.Execute()
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print(f"Could not load DICOM series from directory: {directory_path}")
        return None
    return image

def apply_transformations(patient_directories, labels_directory, dat_file_path, label_filename="labels.mha"):
    label_path = os.path.join(labels_directory, label_filename)
    label_image = sitk.ReadImage(label_path)
    label_to_hu = parse_dat_file(dat_file_path)
    hu_mapped_label_image = map_to_hu_values(label_image, label_to_hu, labels_directory)

    for patient_directory in patient_directories:
        patient_ct_image = load_patient_ct_scan(patient_directory)
        if not patient_ct_image:
            continue

        transform_path = os.path.join(patient_directory, "rigid_transformation.tfm")
        parameters = read_transform_parameters(transform_path)

        if parameters:
            transformed_patient_dir = os.path.join("Results", os.path.basename(patient_directory))
            os.makedirs(transformed_patient_dir, exist_ok=True)

            patient_ct_transformed_name = os.path.basename(patient_directory) + "_patient_ct"
            labels_transformed_name = os.path.basename(patient_directory) + "_phantom_labels"

            apply_rigid_body_transformation(patient_ct_image, parameters, patient_ct_transformed_name, transformed_patient_dir)
            apply_rigid_body_transformation(hu_mapped_label_image, parameters, labels_transformed_name, labels_directory)

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

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations.py [patient_directories] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directories = ["Patient_CT_Scan_1", "Patient_CT_Scan_2", "Patient_CT_Scan_3", "Patient_CT_Scan_4"]
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]
    apply_transformations(patient_directories, labels_directory, dat_file_path)
