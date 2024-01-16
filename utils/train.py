import model, dataset, helpers
from tqdm.auto import tqdm

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
        train_loss = 0
        
        model.train()
            
        for batch, (X, y) in enumerate(train_dataloader):
            # Put on correct device
            X, y = X.to(device), y.to(device)
            
            # Forward Pass
            train_pred = model(X)
            
            # Calculate Loss
            loss = loss_fn(train_pred, y)
            train_loss += loss
            
            # Optimizer zero grad
            optimizer.zero_grad()
            
            # Loss Backward
            loss.backward()
            
            # Optimizer Step
            optimizer.step()
            
        return train_loss
                
        
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
        test_loss = 0
        
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
                
        test_loss /= len(test_dataloader)
        
        return test_loss
                
    def train(self):
        
        results = {'train_loss': [],
                'test_loss': []}
        
        for epoch in tqdm(range(epochs)):
            train_loss = Test.train_step()
            
            test_loss = Test.test_step()
            
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            
            print(f"Epoch: {epoch}  || Train Loss: {train_loss} || Test Loss: {test_loss}")
            
            
        return results


if __name__ == '__main__':
    project_directory = Path('G:/Projects/FSRCNN-2016/')

    zip_data_path = Path(os.path.join(project_directory, 'zips'))
    image_data_path = Path(os.path.join(project_directory, 'images'))

    train_zip_path =  Path(os.path.join(zip_data_path, 'DIV2K_train_HR.zip'))
    train_image_path =  Path(os.path.join(image_data_path, 'train_images'))

    valid_zip_path = Path(os.path.join(zip_data_path, 'DIV2K_valid_HR.zip'))
    valid_image_path = Path(os.path.join(image_data_path, 'valid_images'))

    unzip(train_zip_path, train_image_path)
    unzip(valid_zip_path, valid_image_path)
    
    augmented_image_path = Path(os.path.join(project_directory, 'augmented_images'))
    if augmented_image_path.is_dir():
            print(f"{augmented_image_path} already exists")
    else:
        augmented_image_path.mkdir(parents=True, exist_ok=True)
    
    
    augment_images(in_path=train_image_path,
               out_path=augmented_image_path,
               subclass='train')

    augment_images(in_path=valid_image_path,
                out_path=augmented_image_path,
                subclass='valid')
    
    augmented_image_path_train = Path(os.path.join(augmented_image_path, 'train'))
    augmented_image_path_valid = Path(os.path.join(augmented_image_path, 'valid'))  
    
    
    # DataLoader
    train_data_custom = TrainDIV2K(dir_path=augmented_image_path_train,
                                scale=2)

    valid_data_custom = EvalDIV2K(dir_path=augmented_image_path_valid,
                                scale=2)
    
    BATCH_SIZE = 1
    # NUM_WORKERS = os.cpu_count()

    train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                        batch_size=BATCH_SIZE,
                                        #  num_workers=NUM_WORKERS,
                                        shuffle=True
                                        )

    valid_dataloader_custom = DataLoader(dataset=valid_data_custom,
                                        batch_size=BATCH_SIZE,
                                        #  num_workers=NUM_WORKERS,
                                        shuffle=False)
    
    model_0 = FSRCNN(scale=2)
    
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                     lr=1e-3)
    
    loss_fn = nn.MSELoss()

    Train(model=model_0,
        train_dataloader=train_dataloader_custom,
        test_dataloader=valid_dataloader_custom,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        device=device).train()