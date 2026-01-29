import os
import pickle
from pathlib import Path
from train import model, SCALE, BATCH_SIZE, EPOCHS

if __name__ == '__main__':
    
    # Pickle it
    with open(f'models/FSRCNN_{SCALE}s_{EPOCHS}e_{BATCH_SIZE}b_0.1.0.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'models/FSRCNN_{SCALE}s_{EPOCHS}e_{BATCH_SIZE}b_0.1.0.pickle', 'rb') as f:
        pickle.load(f)
    
    # Save Model
    MODEL_PATH = Path("models")
    if MODEL_PATH.is_dir():
            print(f"{augmented_image_path} already exists")
    else:
        MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)
        
    MODEL_NAME = f'FSRCNN_{SCALE}s_{EPOCHS}e_{BATCH_SIZE}b_0.1.0.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
            f= MODEL_SAVE_PATH)