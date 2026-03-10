import os
import math
import torch
import logging
from torch import nn
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.functional import to_pil_image

# Configure logging
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def upscale(model, in_loc, out_loc):
    logger.info(f"Starting upscale for {in_loc}")
    # image = Image.open('api/static/input/{}'.format(file)).convert('RGB')
    image = Image.open(in_loc).convert('RGB')
    logger.info(f"Image opened: {image.size}")
    
    # Optional: Limit maximum input size to prevent memory issues and speed up processing
    MAX_DIMENSION = 4096  # Adjust based on your needs
    if max(image.size) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized large image to: {image.size}")
    
    image_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image).unsqueeze(0)
    
    # Move to GPU if available
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    logger.info(f"Image tensor moved to {device}")
    
    # Use automatic mixed precision for faster inference (GPU only)
    use_amp = device.type == 'cuda'
    
    with torch.inference_mode():
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                HR_img = model(image_tensor)
        else:
            HR_img = model(image_tensor)
    
    logger.info(f"Model inference complete. Output shape: {HR_img.shape}")

    # Convert to PIL image directly (much faster than matplotlib)
    image_formatted = HR_img.squeeze(0).clamp(0, 1).cpu()
    
    # Convert tensor to PIL - torchvision expects [C, H, W] format with values in [0, 1]
    logger.info(f"Tensor range: min={image_formatted.min():.4f}, max={image_formatted.max():.4f}, shape={image_formatted.shape}")
    pil_image = to_pil_image(image_formatted)
    
    # Ensure RGB mode for compatibility
    if pil_image.mode != 'RGB':
        logger.info(f"Converting from {pil_image.mode} to RGB")
        pil_image = pil_image.convert('RGB')
    
    # Save directly with PIL - detect format from extension
    output_path = f"api/static/output/{out_loc}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use pathlib to get the file extension and determine format
    output_ext = Path(output_path).suffix.lower()
    format_map = {
        '.png': 'PNG',
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.webp': 'WEBP',
        '.bmp': 'BMP'
    }
    save_format = format_map.get(output_ext, 'JPEG')  # Default to JPEG
    
    logger.info(f"Saving as {save_format} to {output_path}")
    
    try:
        if save_format == 'JPEG':
            # Reduced quality from 95 to 85 for faster saving (still excellent quality)
            pil_image.save(output_path, format=save_format, quality=85, optimize=False)
        else:
            pil_image.save(output_path, format=save_format, optimize=False)
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Successfully saved {save_format} image: {output_path}, size: {pil_image.size}, file_size: {file_size} bytes")
        else:
            logger.error(f"File was not created: {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise
    
    return image_formatted




if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    loaded_model_x3 = FSRCNN(scale=3)
    loaded_model_x3.load_state_dict(torch.load(f='api/models/FSRCNN_3s_10e_1b_0.2.0.pth', map_location=device))
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
    HR_pred = upscale(loaded_model_x3, random_img, 'test.png')
    print("Upscaled")
    # print(HR_pred.permute(0,2,3,1).squeeze().shape)
    OG_im = OG_im.permute(1,2,0)
    OG_crop = OG_crop.permute(1,2,0)
    LR_im = LR_im.permute(1,2,0)
    ims = [OG_im, OG_crop, LR_im, HR_pred]
    for image in ims:
        print(image.shape)
        plt.imshow(image)
        plt.axis(False)
        plt.show()