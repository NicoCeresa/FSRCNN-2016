import os
import PIL
import math 
import torch
import pickle
import random 
import zipfile
import pathlib
import numpy as np
from torch import nn
from PIL import Image
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import v2, InterpolationMode


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()


class FSRCNN(nn.Module):
    """
    Model from https://arxiv.org/abs/1608.00367
    """
    def __init__(self, scale:int, num_channels=3, d=56, s=12, m=4):
        """
        d: LR feature dimension
        s: level of shrinking
        m: number of mapping layers
        n: scaling factor
        """
        super(FSRCNN, self).__init__()
        if scale < 2  and scale > 5:
            raise ValueError("Scaling must be 2, 3, or 4")
        else:
            self.scale = scale
        
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                               out_channels=d,
                               kernel_size=5,
                               padding=5//2,
                               padding_mode='zeros',
                               ),
                                   nn.PReLU(d))
        
        self.Conv2 = nn.Sequential(nn.Conv2d(in_channels=d,
                               out_channels=s,
                               kernel_size=1,
                               ),
                                   nn.PReLU(s))
        
        self.Conv3 = []
        for _ in range(m):
            self.Conv3.extend([nn.Conv2d(in_channels=s,
                               out_channels=s,
                               kernel_size=3,
                               padding=3//2,
                               padding_mode='zeros',
                               ),
                                nn.PReLU(s)])
        self.Conv3 = nn.Sequential(*self.Conv3)
        
        self.Conv4 = nn.Sequential(nn.Conv2d(in_channels=s,
                               out_channels=d,
                               kernel_size=1,
                               ),
                                   nn.PReLU(d))
        
        self.DeConv = nn.ConvTranspose2d(in_channels=d,
                                         out_channels=num_channels,
                                         kernel_size=9,
                                         stride=self.scale,
                                         padding=9//2,
                                         output_padding=self.scale-1,
                                         )
        
        self._init_weights()
        
    def _init_weights(self):
        for w in self.Conv1:
            if isinstance(w, nn.Conv2d):
                nn.init.normal_(w.weight.data, mean=0.0, std=math.sqrt(2/(w.out_channels*w.weight.data[0][0].numel())))
        for w in self.Conv2:
            if isinstance(w, nn.Conv2d):
                nn.init.normal_(w.weight.data, mean=0.0, std=math.sqrt(2/(w.out_channels*w.weight.data[0][0].numel())))
        for w in self.Conv3:
            if isinstance(w, nn.Conv2d):
                nn.init.normal_(w.weight.data, mean=0.0, std=math.sqrt(2/(w.out_channels*w.weight.data[0][0].numel())))
        for w in self.Conv4:
            if isinstance(w, nn.Conv2d):
                nn.init.normal_(w.weight.data, mean=0.0, std=math.sqrt(2/(w.out_channels*w.weight.data[0][0].numel())))
        
        nn.init.normal_(self.DeConv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.DeConv.bias.data)
            
        
    def forward(self, x):
        x = self.Conv1(x)
        # print(block1.shape)
        x = self.Conv2(x)
        # print(block2.shape)
        x = self.Conv3(x)
        # print(block3.shape)
        x = self.Conv4(x)
        # print(block4.shape)
        x = self.DeConv(x)
        return x

loaded_model_mse = FSRCNN(scale=3)
loaded_model_mse.load_state_dict(torch.load(f='api/models/FSRCNN_3s_10e_1b_0.2.0.pth', map_location=device))

def reformat(img, crop_size):
    im_height = img.height
    im_width = img.width
    
    img_crop = v2.Compose(
        [v2.ToImage(), 
         v2.CenterCrop(size=crop_size),
         v2.ToDtype(torch.float32, scale=True)])
    
    img_nocrop = v2.Compose(
        [v2.ToImage(), 
         v2.ToDtype(torch.float32, scale=True)])
    return img_crop(img), img_nocrop(img)

def downscale(img, scale:int, crop_size):
    im_height = img.height
    im_width = img.width
    train_img_transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize(size=((int(im_height//scale)), (int(im_width//scale))), interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(size=crop_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)])
    return train_img_transform(img)

def upscale_cropped(model, img, crop_size):
    HR_img = model(img.convert('RGB').unsqueeze(0))
    # im_height = HR_img.height
    # im_width = HR_img.width
    img_transform = v2.Compose(
        [v2.ToImage(), 
         v2.CenterCrop(size=crop_size),
         v2.ToDtype(torch.float32, scale=True)])
    return img_transform(HR_img)

from torchvision.transforms.functional import to_pil_image

# def upscale(model, in_filename, out_filepath):
#     image = Image.open(in_filename).convert('RGB')
#     # Convert the image to a PyTorch tensor and add a batch dimension
#     image_tensor = v2.ToTensor()(image).unsqueeze(0)
#     # Perform the upscale operation using the model
#     with torch.inference_mode():
#         HR_tensor = model(image_tensor).squeeze(0)
#     # Convert the output tensor back to a PIL image
#     HR_img = to_pil_image(HR_tensor)
#     # Save the resulting image to the output file
#     HR_img.save(out_filepath)
    
#     return HR_tensor.cpu()


def upscale(model, in_loc, out_loc):
    # image = Image.open('api/static/input/{}'.format(file)).convert('RGB')
    image = Image.open(in_loc).convert('RGB')
    image_tensor = v2.ToTensor()(image).unsqueeze(0)
    with torch.inference_mode():
        HR_img = model(image_tensor)
    # im_height = HR_img.height
    # im_width = HR_img.width
    image_formatted = HR_img.squeeze(0).type(torch.float32).cpu().permute(1,2,0)
    print(image_formatted.shape)
    
    plt.imshow(image_formatted)
    plt.axis(False)
    plt.margins(False)
    plt.savefig(f"api/static/output/{out_loc}", bbox_inches='tight', pad_inches=0)
    return image_formatted



if __name__ == '__main__':
    project_directory = Path('G:/Projects/FSRCNN-2016/')
    image_data_path = Path(os.path.join(project_directory, 'images'))
    train_image_path =  Path(os.path.join(image_data_path, 'train_images'))
    image_path_list = list(train_image_path.glob("*/*.png"))
    random_img = random.choice(image_path_list)
    print(random_img)
    # img_path = image_path_list[420]
    im = Image.open(random_img)
    
    norm_size = (im.height//5, im.width//5)
    LR_size = ((im.height//3)//5, (im.width//3)//5)    
    
    OG_crop, OG_im = reformat(im, crop_size=norm_size)
    LR_im = downscale(im, scale=3,crop_size=LR_size)
    HR_pred = upscale(loaded_model_mse, random_img, 'test.png')
    print("Upscaled")
    # # print(HR_pred.permute(0,2,3,1).squeeze().shape)
    # OG_im = OG_im.permute(1,2,0)
    # OG_crop = OG_crop.permute(1,2,0)
    # LR_im = LR_im.permute(1,2,0)
    # HR_pred = HR_pred.permute(1,2,0)
    # ims = [OG_im, OG_crop, LR_im, HR_pred]
    # for image in ims:
    #     print(image.shape)
    #     plt.imshow(image)
    #     plt.axis(False)
    #     plt.show()