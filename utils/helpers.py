# Miscellanious Helper Fxns

def unzip(zip_path, image_path):
    """unzips

    Args:
        zip_path (_type_): _description_
        image_path (_type_): _description_
    """
    if Path(image_path).is_dir():
        print(f"{image_path} already exists")
    else:
        image_path.mkdir(parents=True, exist_ok=True)      
        
    if len(os.listdir(image_path)) > 0:
        print("Images already in folder") 
        pass
    else:
        with zipfile.ZipFile(zip_path, 'r') as zip_train:
            print('Unzipping Data')
            zip_train.extractall(image_path)    
            
            
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


import h5py
import numpy as np
from pathlib import Path
import cv2
from glob import glob


def augment_images(in_path, out_path, subclass):
    
    augmented_image_path = Path(os.path.join(project_directory, 'augmented_images'))
    subclass_dir = Path(os.path.join(augmented_image_path, subclass))
    if subclass_dir.is_dir():
        print(f"{augmented_image_path}/{subclass} already exists")
        
    else:
        subclass_dir.mkdir(parents=True, exist_ok=True)
        
        
    if len(os.listdir(subclass_dir)) > 0:
        print(f"files already loaded into {str(out_path)}\{subclass}")
        pass
    
    else:
    
        train_augmented = []
        
        og_train_images = list(glob(os.path.join(in_path, f"*/*.png")))
        for image_path in og_train_images:
            path_str = str(image_path)
            path_name = os.path.basename(path_str).split('.')[0]
            image = cv2.imread(image_path)
            for scale in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for rotation in [0, 90, 180, 270]:
                    augmented_image_path = os.path.join(out_path, subclass, f"{path_name}_{1 if str(scale).split('.')[1] == '0' else str(scale).split('.')[1]}_{rotation}.png")
                    augmented = cv2.resize(src=image, 
                                        dsize=(int(image.shape[0] * scale), 
                                        int(image.shape[1] * scale)), 
                                        interpolation=cv2.INTER_CUBIC)
                    image_center = tuple(np.array(image.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
                    augmented = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_CUBIC)
                    # augmented = augmented.rotate(rotation, expand=True)
                    # train_augmented.append(augmented)
                    cv2.imwrite(augmented_image_path, augmented)
                    
        print(f"Finished Loading Images {out_path}\{subclass}")


def plot_transformation(idx, data, n=3):
    """
    input:
        path_list: list of paths
        transform: a transform function
        n: the number of images to display
    output:
        displays random images in pairs of the original and transformed image
    """
    plt.figure(figsize=(30,20))
    random_images = random.sample(range(len(data)), n)
    for img in random_images:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        images = data.__getitem__(img)
        # OG image
        ax[0].imshow(images[1].squeeze(0))
        ax[0].set_title(f"Original")
        ax[0].axis(False)
        
        # Transformed image
        ax[1].imshow(images[0].squeeze(0))
        ax[1].set_title(f"Downsampled")
        ax[1].axis(False)