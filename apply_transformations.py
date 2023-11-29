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

def read_and_display_tfm_file(transform_path):
    try:
        with open(transform_path, 'r') as tfm_file:
            tfm_contents = tfm_file.read()
        print(f"Contents of {transform_path}:\n{tfm_contents}")
    except FileNotFoundError:
        print(f"Transformation file not found: {transform_path}")
        return None
    return tfm_contents

def apply_transformation(label_image, transform_path, transform_name):
    tfm_contents = read_and_display_tfm_file(transform_path)
    if not tfm_contents:
        return None

    transform = sitk.ReadTransform(transform_path)
    print(f"Transform type: {type(transform).__name__}")
    print(f"Transform parameters: {transform.GetParameters()}")

    # Additional checks for coordinate system alignment (if applicable)
    # ...

    transformed_label = sitk.Resample(label_image, label_image, transform, sitk.sitkNearestNeighbor, 0.0, label_image.GetPixelID())
    return transformed_label

def map_to_hu_values(transformed_label, label_to_hu, transform_name, output_path):
    if transformed_label is None:
        print(f"No transformation applied for {transform_name}. Skipping HU mapping.")
        return

    transformed_label_array = sitk.GetArrayFromImage(transformed_label)
    print(f"Transformed label image range before mapping: {np.min(transformed_label_array)} - {np.max(transformed_label_array)}")

    hu_mapped_array = np.copy(transformed_label_array)
    for label, hu_value in label_to_hu.items():
        hu_mapped_array[transformed_label_array == label] = hu_value

    hu_mapped_image = sitk.GetImageFromArray(hu_mapped_array)
    hu_mapped_image.CopyInformation(transformed_label)

    print(f"Transformed HU-mapped image data type: {hu_mapped_image.GetPixelIDTypeAsString()}")
    print(f"Transformed HU-mapped image range: {np.min(hu_mapped_array)} - {np.max(hu_mapped_array)}")

    transformed_label_path = os.path.join(output_path, f"transformed_hu_mapped_{transform_name}.mha")
    sitk.WriteImage(hu_mapped_image, transformed_label_path)
    print(f"Transformed HU-mapped label image saved at {transformed_label_path}")

def test_transformation(label_image, output_path):
    print("Applying a test transformation (slight translation)")
    translation = sitk.TranslationTransform(3, (1.0, 1.0, 1.0))
    transformed_label = sitk.Resample(label_image, label_image, translation, sitk.sitkNearestNeighbor, 0.0, label_image.GetPixelID())

    transformed_label_path = os.path.join(output_path, "test_transformed.mha")
    sitk.WriteImage(transformed_label, transformed_label_path)
    print(f"Test transformed label image saved at {transformed_label_path}")

def apply_transformations_to_labels(patient_directory, labels_directory, dat_file_path, label_filename="labels.mha"):
    print(f"Reading label image from {labels_directory}")
    label_path = os.path.join(labels_directory, label_filename)
    label_image = sitk.ReadImage(label_path)

    print(f"Parsing .dat file: {dat_file_path}")
    label_to_hu = parse_dat_file(dat_file_path)

    # Test with a known transformation
    test_transformation(label_image, patient_directory)

    transform_name = "full_ct"
    transform_path = os.path.join(patient_directory, "rigid_transformation.tfm")
    transformed_label = apply_transformation(label_image, transform_path, transform_name)
    if transformed_label:
        map_to_hu_values(transformed_label, label_to_hu, transform_name, patient_directory)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python apply_transformations_to_labels.py [patient_directory] [labels_directory] [dat_file_path]")
        sys.exit(1)

    patient_directory = sys.argv[1]
    labels_directory = sys.argv[2]
    dat_file_path = sys.argv[3]
    apply_transformations_to_labels(patient_directory, labels_directory, dat_file_path)
