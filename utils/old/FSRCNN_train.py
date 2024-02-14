import os
import torch
from torch import nn
from pathlib import Path
from model_2 import FSRCNN
from tqdm.auto import tqdm
import torch.optim as optim
from FSRCNN_helpers import calc_psnr
from dataset_2 import TrainDIV2K, EvalDIV2K
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Train():
    
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 device=DEVICE):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device=device


    def train_step(self
        # model: torch.nn.Module,
        # dataloader: torch.utils.data.DataLoader,
        # loss_fn: torch.nn.Module,
        # optimizer: torch.optim.Optimizer,
        # device=DEVICE
               ):
        """
        Trains one batch of images and returns the loss
        """
        # Initialize training loss
        train_loss = 0

        model = self.model
        train_dataloader = self.train_dataloader
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        
        model.train()
            
        for batch, (X, y) in enumerate(train_dataloader):
            # Put on correct device
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward Pass
            train_pred = self.model(X)
            
            # Calculate Loss
            loss = loss_fn(train_pred, y)
            train_loss += loss.item()
            
            # Optimizer zero grad
            optimizer.zero_grad()
            
            # Loss Backward
            loss.backward()
            
            # Optimizer Step
            optimizer.step()
            
        return train_loss
                    
        
    def test_step(self
        # model: torch.nn.Module,
        # dataloader: torch.utils.data.DataLoader,
        # loss_fn: torch.nn.Module,
        # device=DEVICE
        ):
    
        # Put in eval mode
        model.eval()

        # Initialize test loss
        test_loss = 0

        # Put in inference mode
        with torch.inference_mode():
            
            for batch, (X, y) in enumerate(dataloader):
                # Put data on correct device
                X, y = X.to(device), y.to(device)
                
                # Forward Pass
                test_pred = model(X)
                
                # Calculate loss
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                
        test_loss /= len(dataloader)

        return test_loss
                
                
    def train(self
        # model: torch.nn.Module,
        # train_dataloader: torch.utils.data.DataLoader,
        # test_dataloader: torch.utils.data.DataLoader,
        # optimizer: torch.optim.Optimizer,
        # loss_fn: torch.nn.Module,
        # epochs: int
        # device=DEVICE
          ):
    
        results = {'train_loss': [],
                  'test_loss': []}

        for epoch in tqdm(range(self.epochs)):
            print(f"Training epoch {epoch}")
            train_loss = Train(
                            model=self.model,
                            train_dataloader=self.train_dataloader,
                            test_dataloader=self.test_dataloader,
                            optimizer=self.optimizer,
                            loss_fn=self.loss_fn,
                            epochs=self.epochs,
                            device=DEVICE).train_step()
            
            test_loss = test_step(
                    model=self.model,
                    dataloader=self.test_dataloader,
                    loss_fn=self.loss_fn,
                    device=self.device
            )
            
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            
            print(f"Epoch: {epoch}  || Train Loss: {train_loss:.4f} || Test Loss: {test_loss:.4f}")
            
        return results


if __name__ == '__main__':
    SCALE = 2
    EPOCHS = 1
    BATCH_SIZE = 5
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

    Train(model=model,
        train_dataloader=train_dataloader_custom,
        test_dataloader=valid_dataloader_custom,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE).train()
    
    # Save Model
    MODEL_PATH = Path("models")
    if MODEL_PATH.is_dir():
            print(f"{augmented_image_path} already exists")
    else:
        MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)
        
    MODEL_NAME = f'FSRCNN_{n}s_{BATCH_SIZE}b_{EPOCHS}e_0.1.0.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(),
            f= MODEL_SAVE_PATH)

    # Load in save state_dict
    loaded_model = FSRCNN(scale=SCALE).to(DEVICE)
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    loaded_model.to(DEVICE)
    
    psnr_list = []

    # Calculate PSNR
    for idx, (X, y) in tqdm(enumerate(eval_dataloader_custom)):
        X = X.to(DEVICE)
        
        with torch.inference_mode():
            pred = loaded_model(X)
        
        psnr = calc_psnr(pred.cpu(), y.cpu())
        
        psnr_list.append(psnr)

    print(f'Avg PSNR: {np.mean(psnr_list):.4f}\nMax PSNR: {np.max(psnr_list)}')