import math 
import torch
from torch import nn

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
        super().__init__()
        if scale < 2  and scale > 5:
            raise ValueError("Scaling must be 2, 3, or 4")
        else:
            self.scale = scale
        
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                               out_channels=d,
                               kernel_size=5,
                               padding=5//2,
                               padding_mode='zeros',
                               device=device),
                                   nn.PReLU(d))
        
        self.Conv2 = nn.Sequential(nn.Conv2d(in_channels=d,
                               out_channels=s,
                               kernel_size=1,
                               device=device),
                                   nn.PReLU(s))
        
        self.Conv3 = []
        for _ in range(m):
            self.Conv3.extend([nn.Conv2d(in_channels=s,
                               out_channels=s,
                               kernel_size=3,
                               padding=3//2,
                               padding_mode='zeros',
                               device=device),
                                nn.PReLU(s)])
        self.Conv3 = nn.Sequential(*self.Conv3)
        
        self.Conv4 = nn.Sequential(nn.Conv2d(in_channels=s,
                               out_channels=d,
                               kernel_size=1,
                               device=device),
                                   nn.PReLU(d))
        
        self.DeConv = nn.ConvTranspose2d(in_channels=d,
                                         out_channels=num_channels,
                                         kernel_size=9,
                                         stride=self.scale,
                                         padding=9//2,
                                         output_padding=self.scale-1,
                                         device=device)
        
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
        block1 = self.Conv1(x)
        # print(block1.shape)
        block2 = self.Conv2(block1)
        # print(block2.shape)
        block3 = self.Conv3(block2)
        # print(block3.shape)
        block4 = self.Conv4(block3)
        # print(block4.shape)
        out = self.DeConv(block4)
        return out