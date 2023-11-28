import SimpleITK as sitk
import numpy as np
import os
import sys

# Parsing the .dat file to create a mapping from label values to HU values
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

# Applying the mapping to the transformed label image
def apply_and_map_transformation(label_image, transform_path, label_to_hu, output_path, transform_name):
    print(f"Applying transformation: {transform_path}")
    if not os.path.exists(transform_path):
        print(f"Transformation file not found: {transform_path}")
        return

    # Check data type and range before transformation
    print(f"Original label image data type: {label_image.GetPixelIDTypeAsString()}")
    original_array = sitk.GetArrayFromImage(label_image)
    print(f"Original label image range: {np.min(original_array)} - {np.max(original_array)}")

    transform = sitk.ReadTransform(transform_path)
    transformed_label = sitk.Resample(label_image, label_image, transform, sitk.sitkNearestNeighbor, 0.0, label_image.GetPixelID())

    print(f"Mapping transformed labels to HU values for {transform_name}")
    transformed_label_array = sitk.GetArrayFromImage(transformed_label)
    print(f"Transformed label image range before mapping: {np.min(transformed_label_array)} - {np.max(transformed_label_array)}")

    hu_mapped_array = np.copy(transformed_label_array)
    for label, hu_value in label_to_hu.items():
        hu_mapped_array[transformed_label_array == label] = hu_value

    hu_mapped_image = sitk.GetImageFromArray(hu_mapped_array)
    hu_mapped_image.CopyInformation(transformed_label)

    # Check data type and range after mapping
    print(f"Transformed HU-mapped image data type: {hu_mapped_image.GetPixelIDTypeAsString()}")
    print(f"Transformed HU-mapped image range: {np.min(hu_mapped_array)} - {np.max(hu_mapped_array)}")

    transformed_label_path = os.path.join(output_path, f"transformed_hu_mapped_{transform_name}.mha")
    sitk.WriteImage(hu_mapped_image, transformed_label_path)
    print(f"Transformed HU-mapped label image saved at {transformed_label_path}")


def apply_transformations_to_labels(patient_directory, labels_directory, dat_file_path, label_filename="labels.mha"):
    print(f"Reading label image from {labels_directory}")
    label_path = os.path.join(labels_directory, label_filename)
    label_image = sitk.ReadImage(label_path)

    print(f"Parsing .dat file: {dat_file_path}")
    label_to_hu = parse_dat_file(dat_file_path)

    transformations = {
        "rigid_transformation_lung": "lung",
        "rigid_transformation_bone": "bone",
        "rigid_transformation": "full_ct"
    }

    for transform_file, transform_name in transformations.items():
        transform_path = os.path.join(patient_directory, f"{transform_file}.tfm")
        apply_and_map_transformation(label_image, transform_path, label_to_hu, patient_directory, transform_name)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations_to_labels.py [patient_directory] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directory = sys.argv[1]
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]
    apply_transformations_to_labels(patient_directory, labels_directory, dat_file_path)
