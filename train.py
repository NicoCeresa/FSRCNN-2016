import helpers
from model import FSRCNN
from dataset import TrainDIV2K, EvalDIV2K
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn


class Train():
    
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 device=device):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device=device

    def train_step(self, 
                 model=self.model,
                 train_dataloader=self.train_dataloader,
                 optimizer=self.optimizer,
                 loss_fn=self.loss_fn,
                 epochs=self.epochs,
                 device=self.device):
        """
        Trains one batch of images and returns the loss
        """
        # Initialize training loss
        train_loss, train_psnr = 0, 0
        
        model.train()
            
        for batch, (X, y) in enumerate(train_dataloader):
            # Put on correct device
            X, y = X.to(device), y.to(device)
            
            # Forward Pass
            train_pred = model(X)
            
            # Calculate Loss
            loss = loss_fn(train_pred, y)
            train_loss += loss
            
            # Calculate PSNR
            psnr = calc_psnr(y, train_pred)
            train_psnr += psnr
            
            # Optimizer zero grad
            optimizer.zero_grad()
            
            # Loss Backward
            loss.backward()
            
            # Optimizer Step
            optimizer.step()
                    
        return train_loss, train_psnr
                
        
    def test_step(self, 
                 model=self.model,
                 test_dataloader=self.test_dataloader,
                 optimizer=self.optimizer,
                 loss_fn=self.loss_fn,
                 epochs=self.epochs,
                 device=self.device):
        # Put in eval mode
        model.eval()
        
        # Initialize test loss
        test_loss, test_psnr = 0, 0
        
        # Put in inference mode
        with torch.inference_mode():
            
            for batch, (X, y) in enumerate(test_dataloader):
                # Put data on correct device
                X, y = X.to(device), y.to(device)
                
                # Forward Pass
                test_pred = model(X)
                
                # Calculate loss
                loss = loss_fn(test_pred, y)
                test_loss += loss
                
                # Calculate PSNR
                psnr = calc_psnr(y, test_pred)
                test_psnr += psnr
                
        test_loss /= len(test_dataloader)
        test_psnr /= len(test_dataloader)
        
        return test_loss, test_psnr
                
                
    def train(self):
        
        results = {'train_loss': [],
                'test_loss': [],
                'train_psnr':[],
                'test_psnr':[]}
        
        print("Training...")
        for epoch in tqdm(range(epochs)):
            train_loss, train_psnr = Train.train_step()
            
            test_loss, test_psnr = Train.test_step()
            
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            results['train_psnr'].append(train_psnr)
            results['test_psnr'].append(test_psnr)
            
            print(f"Epoch: {epoch}  || Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.4f} || Test Loss: {test_loss:.4f} | Test PSNR: {test_psnr}")
            
        return results


if __name__ == '__main__':
    EPOCHS = 1
    BATCH_SIZE = 1
    SCALE = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # NUM_WORKERS = os.cpu_count()
    augmented_image_path = 'augmented_images'
    
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
                                        #  num_workers=NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True
                                        )

    valid_dataloader_custom = DataLoader(dataset=valid_data_custom,
                                        batch_size=BATCH_SIZE,
                                        #  num_workers=NUM_WORKERS,
                                        shuffle=False,
                                        pin_memory=True)
    
    model = FSRCNN(scale=SCALE).to(device)
    
    optimizer = torch.optim.SGD(params=model.parameters(),
                     lr=1e-3)
    
    loss_fn = nn.MSELoss()

    Train(model=model,
        train_dataloader=train_dataloader_custom,
        test_dataloader=valid_dataloader_custom,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device).train()