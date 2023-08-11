# In[1]:
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from voxelmorph.torch.networks import VxmDense
from models import VxmDense
from CTScanDataset import CTScanDataset, CTScanPairDataset

# In[2]:
# TrainingDNN.py
def train_dnn_model(output_path, 
                    phantom_directory_path, 
                    patient_directory_path, 
                    mask_name, # New parameter for mask type (e.g., "lung" or "bone")
                    num_epochs=10,
                    model_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]],
                    learning_rate=0.001,
                    save_best=True):
    
    print(f"Training DNN model for {mask_name} mask...")
    
    # Initialize the model and optimizer
    model = VxmDense(inshape=(256, 256, 256), nb_unet_features=model_features)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Load the phantom and patient CT scans
    print(f"Loading {mask_name} CT scan datasets...")
    phantom_dataset = CTScanDataset(phantom_directory_path)
    patient_dataset = CTScanDataset(patient_directory_path)

    # Pair up the phantom and patient scans
    print(f"Pairing up the {mask_name} phantom and patient scans...")
    pairs = list(zip(phantom_dataset, patient_dataset))
    paired_dataset = CTScanPairDataset(pairs)

    best_loss = float('inf')

    # Train the model
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1} for {mask_name} mask')
        epoch_loss = 0.0
        
        for i, (fixed_image, moving_image) in tqdm(enumerate(paired_dataset), total=len(paired_dataset)):
            optimizer.zero_grad()
            y_pred, _ = model(moving_image.float().unsqueeze(0), fixed_image.float().unsqueeze(0))
            loss = criterion(y_pred, fixed_image.float().unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(paired_dataset)
        
        print(f'Epoch {epoch + 1} - Loss: {epoch_loss} for {mask_name} mask')
        
        if save_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            model_filename = f"best_{mask_name}_model.pth"  # Modify filename based on mask type
            print(f"Saving the best {mask_name} model so far...")
            torch.save(model.state_dict(), os.path.join(output_path, model_filename))

    if not save_best:
        model_filename = f"{mask_name}_model.pth"  # Modify filename based on mask type
        print(f"Saving the trained {mask_name} model...")
        torch.save(model.state_dict(), os.path.join(output_path, model_filename))
        
    print(f"Training completed for {mask_name} mask.")

