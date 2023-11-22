import SimpleITK as sitk
import os
import sys


def apply_transformations_to_labels(patient_directory, labels_directory, label_filename="labels.mha"):
    """
    Apply specified transformations to the labels file.

    Args:
        patient_directory (str): Directory containing patient-specific transformation files.
        labels_directory (str): Directory containing the label file.
        label_filename (str): Filename of the label image to be transformed (default "labels.mha").
    """
    label_path = os.path.join(labels_directory, label_filename)
    # Read the label image
    label_image = sitk.ReadImage(label_path)

    # Define transformation file names
    transformation_filenames = [
        "rigid_transformation_lung.tfm",
        "rigid_transformation_bone.tfm",
        "rigid_registration.tfm"
    ]

    for transform_filename in transformation_filenames:
        transform_path = os.path.join(patient_directory, transform_filename)

        # Check if the transformation file exists
        if not os.path.exists(transform_path):
            print(f"Transformation file not found: {transform_path}")
            continue

        # Read the transformation
        transform = sitk.ReadTransform(transform_path)

        # Apply the transformation
        transformed_label = sitk.Resample(label_image, label_image, transform,
                                          sitk.sitkNearestNeighbor, 0.0, label_image.GetPixelID())

        # Update label_image for next transformation
        label_image = transformed_label

    # Save the final transformed label image
    transformed_label_path = os.path.join(patient_directory, "transformed_labels.mha")
    sitk.WriteImage(label_image, transformed_label_path)
    print(f"Transformed label image saved at {transformed_label_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply_transformations_to_labels.py [patient_directory] [labels_directory]")
        sys.exit(1)

    patient_directory = sys.argv[1]
    labels_directory = sys.argv[2]
    apply_transformations_to_labels(patient_directory, labels_directory)
