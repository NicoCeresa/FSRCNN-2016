# dataset and dataloader goes here
import torch
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import pathlib
from pathlib import Path
from PIL import Image


class TrainDIV2K(Dataset):
    
    def __init__(self, dir_path:str, scale:int):
        super().__init__()
        # get images from a path
        self.paths = list(pathlib.Path(dir_path).glob(f"*.png"))
        self.scale = scale

    def load_image(self, idx:int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path).convert('RGB')
    
    def train_transform(self, image):
        im_height = (image.height // self.scale) * self.scale
        im_width = (image.width // self.scale) * self.scale
        train_img_transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize(size=((int(im_height//self.scale)), (int(im_width//self.scale))), interpolation=InterpolationMode.BICUBIC),
            v2.Grayscale(num_output_channels=1),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])])
        
        hr_img_transform = v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.Resize(size=(im_height, im_width), interpolation=InterpolationMode.BICUBIC),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ])
        
        return train_img_transform(image), hr_img_transform(image)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the LR and HR images given an index

        Args:
            idx (int): Index of image

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns images in forms of tensors
        """
        img = self.load_image(idx)
        LR, HR = self.train_transform(img)
        return torch.Tensor(np.expand_dims(LR / 255., 0)).squeeze(0), torch.Tensor(np.expand_dims(HR / 255., 0)).squeeze(0)


class EvalDIV2K(Dataset):
    
    def __init__(self, dir_path:str, scale:int):
        super().__init__()
        # get images from a path
        self.paths = list(pathlib.Path(dir_path).glob(f"*.png"))
        self.scale = scale

    def load_image(self, idx:int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path).convert('RGB')
    
    def test_transform(self, image):
        im_height = (image.height // self.scale) * self.scale
        im_width = (image.width // self.scale) * self.scale
        test_img_transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize(size=((int(im_height//self.scale)), (int(im_width//self.scale))), interpolation=InterpolationMode.BICUBIC),
            v2.Grayscale(num_output_channels=1),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ])
        
        hr_img_transform = v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.Resize(size=(im_height, im_width), interpolation=InterpolationMode.BICUBIC),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ])
        
        return test_img_transform(image), hr_img_transform(image)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the LR and HR images given an index

        Args:
            idx (int): Index of image

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns images in forms of tensors
        """
        img = self.load_image(idx)
        LR, HR = self.test_transform(img)
        return torch.Tensor(np.expand_dims(LR / 255., 0)).squeeze(0), torch.Tensor(np.expand_dims(HR / 255., 0)).squeeze(0)
    
if __name__ == '__main__':
    PROJECT_PATH = 'G:/Projects/FSRCNN-2016/'
    project_directory = Path(PROJECT_PATH)

    zip_data_path = Path(os.path.join(project_directory, 'zips'))
    image_data_path = Path(os.path.join(project_directory, 'images'))

    train_zip_path =  Path(os.path.join(zip_data_path, 'DIV2K_train_HR.zip'))
    train_image_path =  Path(os.path.join(image_data_path, 'train_images'))

    valid_zip_path = Path(os.path.join(zip_data_path, 'DIV2K_valid_HR.zip'))
    valid_image_path = Path(os.path.join(image_data_path, 'valid_images'))
    
    # Unzip images to location in env
    unzip(train_zip_path, train_image_path)
    unzip(valid_zip_path, valid_image_path)
    
    # Put Augmented (rotated and scaled) images in diff folder
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