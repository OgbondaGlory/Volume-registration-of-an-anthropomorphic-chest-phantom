# In[1]:
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from voxelmorph.torch.networks import VxmDense
from scipy.ndimage import map_coordinates
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from CTScanDataset import CTScanPairDataset, CTScanDataset
from TrainingDNN import train_dnn_model
from utils import *

# DNNRegistration.py
# In[3]:

#Assuming that you want to pair every combination of images in the dataset, not just consecutive ones
# from itertools import combinations

# def load_data_for_training(directory_path):
#     dataset = CTScanDataset(directory_path)
#     pairs = [pair for pair in combinations(dataset, 2)]
#     return CTScanPairDataset(pairs)


# random pairing
# import random

# def load_data_for_training(directory_path):
#     dataset = CTScanDataset(directory_path)
#     random.shuffle(dataset) # Shuffle the images
#     pairs = [(dataset[i], dataset[i + 1]) for i in range(len(dataset) - 1)]
#     return CTScanPairDataset(pairs)


# Helper functions
def compute_similarity(image1, image2, method='correlation'):
    image1_np = sitk.GetArrayFromImage(image1)
    image2_np = sitk.GetArrayFromImage(image2)

    if method == 'correlation':
        similarity = np.corrcoef(image1_np.flat, image2_np.flat)[0, 1] # correlation coefficient
    elif method == 'ssim':
        similarity = ssim(image1_np, image2_np)
    elif method == 'mse':
        similarity = -mean_squared_error(image1_np, image2_np) # MSE is a distance, so negate it to make it a similarity
    else:
        raise ValueError(f'Unknown method: {method}')

    return similarity


def load_data_for_training(directory_path, method='correlation'):
    # Load the images
    dataset = CTScanDataset(directory_path)

    # Compute the similarity between every pair of images
    similarities = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(i, len(dataset)):
            if i != j:
                similarity = compute_similarity(dataset[i], dataset[j], method=method)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

    # Pair each image with the most similar other image
    pairs = []
    for i in range(len(dataset)):
        j = np.argmax(similarities[i])
        pairs.append((dataset[i], dataset[j]))

    return CTScanPairDataset(pairs)



def preprocess(image_array):
    # Ensure the output is (channel, depth, height, width)
    image_array = np.expand_dims(image_array, axis=0)
    image_tensor = torch.from_numpy(image_array)
    return image_tensor

def apply_displacement_field(moving_image_array, displacement_field):
    coords = np.mgrid[0:moving_image_array.shape[0], 0:moving_image_array.shape[1], 0:moving_image_array.shape[2]]
    coords += displacement_field
    warped_moving_image_array = map_coordinates(moving_image_array, coords, order=3)
    return warped_moving_image_array



# In[3]:
def apply_dnn_registration(output_path, phantom_directory_path, patient_directory_path, mask_name, do_training=True):
    print(f"Starting DNN Registration for {mask_name} mask...")
    
    model_file_path = os.path.join(output_path, f"{mask_name}_model.pth")
    
    if os.path.exists(model_file_path) and not do_training:
        # Load the pre-trained model
        print(f"Loading the pre-trained {mask_name} model...")
        model = VxmDense(inshape=(256, 256, 256), nb_unet_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]])
        model.load_state_dict(torch.load(model_file_path))
    else:
        # Train the model if not already trained and save it
        print(f"Training the DNN model for {mask_name} mask...")
        model = train_dnn_model(output_path, phantom_directory_path, patient_directory_path, mask_name)

    # Load the phantom and patient CT scans
    print("Loading CT scan datasets...")
    phantom_dataset = CTScanDataset(phantom_directory_path)
    patient_dataset = CTScanDataset(patient_directory_path)

    # Pair up the phantom and patient scans
    print("Pairing up the phantom and patient scans...")
    pairs = list(zip(phantom_dataset, patient_dataset))
    paired_dataset = CTScanPairDataset(pairs)

    # Apply the model to a pair of images for visualization
    print("Applying DNN registration to the paired dataset...")
    fixed_image_tensor, moving_image_tensor = paired_dataset[0]  # Change index if necessary

    # Compute the displacement field
    print("Computing the displacement field...")
    displacement_field, _ = model([fixed_image_tensor.float().unsqueeze(0), moving_image_tensor.float().unsqueeze(0)])

    # Convert displacement field back to numpy
    displacement_field = displacement_field.detach().numpy()

    # Apply displacement field to moving image
    print("Applying the displacement field to the moving image...")
    warped_moving_image_array = apply_displacement_field(moving_image_tensor.numpy(), displacement_field)

    # Save the transformed moving image
    print("Saving the transformed moving image...")
    # Save the transformed moving image with mask naming convention
    output_transformed_image_path = os.path.join(output_path, f"transformed_moving_image_{mask_name}.mha")
    sitk.WriteImage(sitk.GetImageFromArray(warped_moving_image_array.squeeze()), output_transformed_image_path)

    # Extracting ISO Surfaces for CNNS
    # Set paths
    print("Extracting ISO surfaces...")
    output_warped_image_path = os.path.join(output_path, "warped_image.mha")
    output_iso_surface_file_path = os.path.join(output_path, "iso_surface_warped.stl")

    # # Save the warped image
    # sitk.WriteImage(sitk.GetImageFromArray(warped_moving_image_array.squeeze()), output_warped_image_path)

    # Load the image
    warped_image = sitk.ReadImage(output_warped_image_path)

    # Extract the iso-surface
    verts, faces = extract_iso_surface(warped_image, level=0.5, smooth=0.0) # you may need to adjust the level and smooth values

    # Save the iso-surface as STL
    save_iso_surface(verts, faces, output_iso_surface_file_path)

    # Display the images after transformation
    print("Displaying the images after transformation...")
    display_images(fixed_image_tensor.numpy().squeeze())
    display_images(warped_moving_image_array.squeeze())
    
    print(f"DNN Registration completed for {mask_name} mask.")
        
    return sitk.ReadImage(output_transformed_image_path)
