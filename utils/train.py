import os
import cv2
import torch
import pickle
import numpy as np
from torch import nn
from pathlib import Path
from models import FSRCNN
from tqdm.auto import tqdm
import torch.optim as optim
from helpers import calc_psnr
from datasets import TrainDIV2K, EvalDIV2K
from torch.utils.data import DataLoader



if __name__ == '__main__':
    SCALE = 2
    EPOCHS = 1
    BATCH_SIZE = 1
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    PROJECT_PATH = 'G:/Projects/FSRCNN-2016/'
    
    project_directory = Path(PROJECT_PATH)
    augmented_image_path = Path(os.path.join(project_directory, 'augmented_images'))
    augmented_image_path_train = Path(os.path.join(augmented_image_path, 'train'))
    augmented_image_path_valid = Path(os.path.join(augmented_image_path, 'valid'))
        
    # Initialize Dataset
    train_data_custom = TrainDIV2K(dir_path=augmented_image_path_train,
                                scale=SCALE)

    valid_data_custom = EvalDIV2K(dir_path=augmented_image_path_valid,
                                scale=SCALE)
    
    # Create DataLoaders
    train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True
                                        )

    valid_dataloader_custom = DataLoader(dataset=valid_data_custom,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    model = FSRCNN(scale=SCALE).to(DEVICE)
    optimizer = torch.optim.SGD(params=model.parameters(),
                     lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Initialize training loss
    train_loss = 0
  
    results = {'train_loss': [],
                'test_loss': []}

    for epoch in tqdm(range(EPOCHS)):
        print(f"\nTraining epoch {epoch}")
        model.train()
        
        for batch, (X, y) in enumerate(train_dataloader_custom):
            # Put on correct device
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Forward Pass
            train_pred = model(X)
            
            # Calculate Loss
            loss = loss_fn(train_pred, y)
            train_loss += loss.item()
            
            # Optimizer zero grad
            optimizer.zero_grad()
            
            # Loss Backward
            loss.backward()
            
            # Optimizer Step
            optimizer.step()
            
        train_loss /= len(train_dataloader_custom)
        results['train_loss'].append(train_loss)
        
        model.eval()

        # Initialize test loss
        test_loss = 0
        psnr_list = []
        # Put in inference mode
        with torch.inference_mode():
            for X, y in valid_dataloader_custom:
                # Put data on correct device
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                # Forward Pass
                test_pred = model(X)
                
                # Calculate loss
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                
                # Calculate PSNR
                psnr = calc_psnr(test_pred.cpu().numpy(), y.cpu().numpy())
                psnr_list.append(psnr)
                
            test_loss /= len(valid_dataloader_custom)
            results['test_loss'].append(test_loss)
        
        print(f"Epoch: {epoch}  || Train Loss: {train_loss:.4f} || Test Loss: {test_loss:.4f} || AVG PSNR: {np.mean(psnr_list):.4f}")

    # Pickle it
    with open(f'app/api/models/FSRCNN_{SCALE}s_{BATCH_SIZE}b_{EPOCHS}e_0.1.0.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'app/api/models/FSRCNN_{SCALE}s_{BATCH_SIZE}b_{EPOCHS}e_0.1.0.pickle', 'rb') as f:
        pickle.load(f)
    
    # Save Model
    MODEL_PATH = Path("app/models")
    if MODEL_PATH.is_dir():
            print(f"{MODEL_PATH} already exists")
    else:
        MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)
        
    MODEL_NAME = f'FSRCNN_{SCALE}s_{BATCH_SIZE}b_{EPOCHS}e_0.1.0.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
            f= MODEL_SAVE_PATH)