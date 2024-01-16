import model, dataset

from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):
    """
    Trains one batch of images and returns the loss
    """
    # Initialize training loss
    train_loss = 0
    
    model.train()
        
    for batch, (X, y) in enumerate(dataloader):
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
            
    
def test_step(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            device=device):

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
            test_loss += loss
            
    test_loss /= len(dataloader)
    
    return test_loss
            
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device=device):
    
    results = {'train_loss': [],
               'test_loss': []}
    
    for epoch in tqdm(range(epochs)):
          train_loss = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
          )
          
          test_loss = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device
          )
          
          results['train_loss'].append(train_loss)
          results['test_loss'].append(test_loss)
          
          print(f"Epoch: {epoch}  || Train Loss: {train_loss} || Test Loss: {test_loss}")
          
          
    return results


if __name__ == '__main__':
    augmented_image_path = Path(os.path.join(project_directory, 'augmented_images'))
    augmented_image_path_train = Path(os.path.join(augmented_image_path, 'train'))
    augmented_image_path_valid = Path(os.path.join(augmented_image_path, 'valid'))  
    
    # DataLoader
    train_data_custom = TrainDIV2K(dir_path=augmented_image_path_train,
                                scale=2)

    valid_data_custom = EvalDIV2K(dir_path=augmented_image_path_valid,
                                scale=2)
    
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                     lr=1e-3)
    
    loss_fn = nn.MSELoss()

    train(model=model_0,
        train_dataloader=train_dataloader_custom,
        test_dataloader=train_dataloader_custom,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        device=device)