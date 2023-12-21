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

def apply_rigid_body_transformation(image, parameters, transform_name, output_path):
    transform = sitk.Euler3DTransform()
    transform.SetParameters(parameters)
    transformed_image = sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor, 0.0, image.GetPixelID())
    save_transformed_image(transformed_image, transform_name, output_path)
    return transformed_image

def save_transformed_image(transformed_image, transform_name, output_path):
    transformed_image_path = os.path.join(output_path, f"{transform_name}.mha")
    sitk.WriteImage(transformed_image, transformed_image_path)
    print(f"Transformed image saved at {transformed_image_path}")

def print_intensity_range(image, description):
    array = sitk.GetArrayFromImage(image)
    print(f"{description} - Intensity Range: {array.min()} to {array.max()}")

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
    reader.SetFileNames(dicom_names)
    try:
        image = reader.Execute()
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print(f"Could not load DICOM series from directory: {directory_path}")
        return None
    return image

def apply_transformations(patient_directories, labels_directory, dat_file_path, label_filename="labels.mha"):
    original_label_image = sitk.ReadImage(os.path.join(labels_directory, label_filename))
    label_to_hu = parse_dat_file(dat_file_path)
    hu_mapped_label_image = map_to_hu_values(original_label_image, label_to_hu)

    # Transformation parameters for each patient
    transformation_parameters = {
        "Patient_CT_Scan_1": [0.07053978264471022, 0.14542691932247254, -0.200145747020965, 12.181300043894305, -266.09215191725474, -369.6768396148839],
        "Patient_CT_Scan_2": [0.17914905925900046, 0.07360253292956428, 0.051284094527928314, 12.183983363082502, -266.0828579001763, -369.6765028701566],
        "Patient_CT_Scan_3": [-0.03354826985625253, 0.001676890142388851, -0.13090946880232018, 12.184485614354324, -266.07540718416936, -369.6741055482845],
        "Patient_CT_Scan_4": [0.15152552307569356, 0.01593586177648574, 0.0823151630224306, -17.205840143098794, -134.32624521176442, -1909.2594367303343]
    }

    for patient_directory in patient_directories:
        patient_ct_image = load_patient_ct_scan(patient_directory)
        if not patient_ct_image:
            continue

        patient_basename = os.path.basename(patient_directory)
        transformed_patient_dir = os.path.join("Results", patient_basename)
        transformed_labels_name = f"{patient_basename}_labels_transformed"

        os.makedirs(transformed_patient_dir, exist_ok=True)

        print(f"Applying transformations for {patient_basename}")

        # Apply transformation to patient CT scan
        transformed_patient_ct = apply_rigid_body_transformation(patient_ct_image, transformation_parameters[patient_basename], f"{patient_basename}_patient_ct_transformed", transformed_patient_dir)
        print_intensity_range(transformed_patient_ct, f"{patient_basename} Patient CT Transformed")
        
        # Apply transformation to HU-mapped label image
        transformed_labels = apply_rigid_body_transformation(hu_mapped_label_image, transformation_parameters[patient_basename], transformed_labels_name, labels_directory)
        print_intensity_range(transformed_labels, f"{patient_basename} Labels Transformed")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations.py [patient_directories] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directories = ["Patient_CT_Scan_1", "Patient_CT_Scan_2", "Patient_CT_Scan_3", "Patient_CT_Scan_4"]
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]
    apply_transformations(patient_directories, labels_directory, dat_file_path)
