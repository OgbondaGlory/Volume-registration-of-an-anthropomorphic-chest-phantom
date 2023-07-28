import numpy as np
import SimpleITK as sitk
from stl import mesh
import matplotlib.pyplot as plt



def segment_lung(image):
    # Apply a threshold to separate lung pixels from others
    thresh_filter = sitk.ThresholdImageFilter()
    thresh_filter.SetLower(-1000)
    thresh_filter.SetUpper(-400)
    thresh_img = thresh_filter.Execute(image)

    # Apply morphological opening
    morph_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    morph_img = morph_filter.Execute(thresh_img)

    # Apply Connected Component Labeling
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(morph_img)

    # Get the two largest components (assumed to be the lungs)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_img = relabel_filter.Execute(cc_img)

    # Get labels array
    labels_array = sitk.GetArrayFromImage(relabel_img)

    # Count the number of pixels in each region and sort by size
    unique, counts = np.unique(labels_array, return_counts=True)
    sorted_counts = sorted(zip(counts, unique), reverse=True)

    # Keep the two largest components (excluding background)
    lung_mask = np.isin(labels_array, [label for _, label in sorted_counts[1:3]])

    # Convert to SimpleITK image for further processing
    lung_mask_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
    lung_mask_sitk.CopyInformation(image)

    return lung_mask_sitk


def segment_bones(image):
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Use thresholding to identify the bones
    # Bones have higher HU than soft tissues or air
    # This range might need adjustment depending on the specific scan
    bone_threshold = 300
    bone_mask = image_array > bone_threshold

    # Convert to SimpleITK image for further processing
    bone_mask_sitk = sitk.GetImageFromArray(bone_mask.astype(np.uint8))
    bone_mask_sitk.CopyInformation(image)

    return bone_mask_sitk


def load_dicom_series(directory_path):
    reader = sitk.ImageSeriesReader()
#     print(directory_path)
#     dicom_names = files = os.listdir(directory_path).sort()
#     reader.SetFileNames(dicom_names)
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
#     print(dicom_names)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def display_images(image, title, montage_slices=10):
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Normalize to 0-255
    image_array = ((image_array - image_array.min()) * (1/(image_array.max() - image_array.min()) * 255)).astype('uint8')
    # Montage of slices using matplotlib
    fig = plt.figure(figsize=(10, 2))
    fig.suptitle(title + " - Slice View")
    slice_interval = image_array.shape[0] // montage_slices
    for i in range(montage_slices):
        ax = fig.add_subplot(1, montage_slices, i + 1)
        ax.imshow(image_array[i * slice_interval], cmap='gray')
        plt.axis('off')
    plt.show()


# Define a simple callback which allows us to monitor registration progress.
def iteration_callback(filter):
    print('\r{0:.2f}'.format(filter.GetMetricValue()), end='')



def generate_checkerboard(fixed_image, moving_image, pattern=(5,5,5)):
    checkerboard_filter = sitk.CheckerBoardImageFilter()
    checkerboard_filter.SetCheckerPattern(pattern)
    checker_image = checkerboard_filter.Execute(fixed_image, moving_image)
    return checker_image

def convert_to_sitk(image_array, original_image):
    # Get the metadata from the original image
    spacing = original_image.GetSpacing()
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()

    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    return sitk_image

def extract_iso_surface(image, level, smooth=0.0):
    # Convert SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Set the isosurface extraction parameters
    iso_surface_extractor = sitk.MarchingCubes()
    iso_surface_extractor.SetBackgroundValue(0)
    iso_surface_extractor.SetValue(1, level)
    iso_surface_extractor.SetInput(sitk.GetImageFromArray(image_array))

    # Extract iso-surfaces
    verts, faces = iso_surface_extractor.Execute()

    return verts, faces

def save_iso_surface(verts, faces, filename):
    # Create a mesh
    iso_surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # Populate the mesh with data
    for i, f in enumerate(faces):
        for j in range(3):
            iso_surface_mesh.vectors[i][j] = verts[f[j],:]

    # Save the mesh to file
    iso_surface_mesh.save(filename)
