# In[1]:
import torch
from torch.optim import Adam
from voxelmorph.torch.networks import VxmDense
from models import VxmDense
from CTScanDataset import CTScanDataset, CTScanPairDataset

# In[2]:
# TrainingDNN.py

def train_dnn_model(output_path, phantom_directory_path, patient_directory_path):
    print("Training DNN model...")
    # Initialize the model and optimizer
    model = VxmDense(inshape=(256, 256, 256), nb_unet_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]])
    optimizer = Adam(model.parameters())

    # Load the phantom and patient CT scans
    print("Loading CT scan datasets...")
    phantom_dataset = CTScanDataset(phantom_directory_path)
    patient_dataset = CTScanDataset(patient_directory_path)

    # Pair up the phantom and patient scans
    print("Pairing up the phantom and patient scans...")
    pairs = list(zip(phantom_dataset, patient_dataset))
    paired_dataset = CTScanPairDataset(pairs)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}')
        for i, (fixed_image, moving_image) in enumerate(paired_dataset):
            optimizer.zero_grad()
            y_pred, _ = model(moving_image.float().unsqueeze(0), fixed_image.float().unsqueeze(0))
            loss = torch.nn.MSELoss()(y_pred, fixed_image.float().unsqueeze(0))
            loss.backward()
            optimizer.step()

    # Save the model
    print("Saving the trained model...")
    torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
    print("Training completed.")
