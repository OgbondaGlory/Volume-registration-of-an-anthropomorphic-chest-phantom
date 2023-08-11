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
                    num_epochs=10,
                    model_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]],
                    learning_rate=0.001,
                    save_best=True):
    
    print("Training DNN model...")
    
    # Initialize the model and optimizer
    model = VxmDense(inshape=(256, 256, 256), nb_unet_features=model_features)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Load the phantom and patient CT scans
    print("Loading CT scan datasets...")
    phantom_dataset = CTScanDataset(phantom_directory_path)
    patient_dataset = CTScanDataset(patient_directory_path)

    # Consider splitting datasets into training and validation subsets here 

    # Pair up the phantom and patient scans
    print("Pairing up the phantom and patient scans...")
    pairs = list(zip(phantom_dataset, patient_dataset))
    paired_dataset = CTScanPairDataset(pairs)

    best_loss = float('inf')

    # Train the model
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        epoch_loss = 0.0
        
        # Consider using tqdm for a progress bar
        for i, (fixed_image, moving_image) in tqdm(enumerate(paired_dataset), total=len(paired_dataset)):
            optimizer.zero_grad()
            y_pred, _ = model(moving_image.float().unsqueeze(0), fixed_image.float().unsqueeze(0))
            loss = criterion(y_pred, fixed_image.float().unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(paired_dataset)
        
        # Consider adding validation loss computation here

        print(f'Epoch {epoch + 1} - Loss: {epoch_loss}')
        
        # Save the best model based on validation performance
        if save_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("Saving the best model so far...")
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))

    if not save_best:
        # Save the model after all epochs
        print("Saving the trained model...")
        torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
        
    print("Training completed.")
