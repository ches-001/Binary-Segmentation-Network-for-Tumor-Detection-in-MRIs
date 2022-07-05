import os
import torch.nn as nn
import torch, torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from .transforms import image_resize


MODEL_PARAM_DIR = r'model_params'
MODEL_PARAM_NAME = r'tumor_segmentation_params.pth.tar'

MODEL_PARAM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(MODEL_PARAM_DIR, MODEL_PARAM_NAME))


class SpatialEncoder(ResNet):
    def __init__(self, input_channels, dropout=0.2, 
                 pretrained=False, block=BasicBlock, block_layers=[3, 4, 6, 3]):
      
        self.block = block
        self.block_layers = block_layers
        
        super(SpatialEncoder, self).__init__(self.block, self.block_layers)
        
        self.input_channels = input_channels
        self.dropout = dropout
        self.pretrained = pretrained
        
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x):
        fmap1 = self.conv1(x)
        x = self.conv2(fmap1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        fmap2 = self.layer1(x)
        fmap3 = self.layer2(fmap2)
        fmap4 = self.layer3(fmap3)
        fmap5 = self.layer4(fmap4)
        
        return fmap1, fmap2, fmap3, fmap4, fmap5


class Decoder(nn.Module):
    def __init__(self, last_fmap_channel, output_channel, n_classes, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.last_fmap_channel = last_fmap_channel
        self.output_channel = output_channel
        self.n_classes = n_classes
        self.dropout = dropout
        
        fmap1_ch, fmap2_ch, fmap3_ch, fmap4_ch, fmap5_ch = (
            self.last_fmap_channel//8,
            self.last_fmap_channel//8, 
            self.last_fmap_channel//4, 
            self.last_fmap_channel//2, 
            self.last_fmap_channel
        )

        if self.last_fmap_channel == 2048:
            fmap1_ch = self.last_fmap_channel//32
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(fmap5_ch, fmap5_ch, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(fmap5_ch),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )
        
        in_ch = fmap5_ch+fmap4_ch
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, fmap5_ch, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(fmap5_ch),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )
        
        in_ch = fmap5_ch+fmap3_ch
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, fmap5_ch, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(fmap5_ch),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )
        
        in_ch = fmap5_ch+fmap2_ch
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, fmap5_ch, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(fmap5_ch),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )
        
        in_ch = fmap5_ch+fmap1_ch
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, self.output_channel*self.n_classes, (2, 2), stride=(2, 2)),
            nn.Sigmoid()
        )
        
    def forward(self, fmap1, fmap2, fmap3, fmap4, fmap5):
        output = self.layer1(fmap5)
        output = torch.cat((output, fmap4), dim=1)
        
        output = self.layer2(output)
        output = torch.cat((output, fmap3), dim=1)
        
        output = self.layer3(output)
        output = torch.cat((output, fmap2), dim=1)
        
        output = self.layer4(output)
        output = torch.cat((output, fmap1), dim=1)
        output = self.layer5(output)
        
        return output


class SegmentNet(nn.Module):
    def __init__(self, input_channels, last_fmap_channel, 
                output_channel, n_classes, 
                pretrained=False, enc_dropout=0.1,
                dec_dropout=0.1, device='cpu'):
        
        super(SegmentNet, self).__init__()
    
        self.input_channels = input_channels
        self.last_fmap_channel = last_fmap_channel
        self.output_channel = output_channel
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.device = device
        
        self.encoder = SpatialEncoder(
            self.input_channels, 
            self.enc_dropout)
        
        self.decoder = Decoder(
            self.last_fmap_channel, 
            self.output_channel, 
            self.n_classes, 
            self.dec_dropout
        )
        
        self.to(self.device)

        if self.pretrained:
            state_dict = torch.load(MODEL_PARAM_PATH, map_location=self.device)
            self.load_state_dict(state_dict['segmentation_net_params'])
    
    def forward(self, x, output_size=None):
        fmap1, fmap2, fmap3, fmap4, fmap5 = self.encoder(x)
        segmentation_mask = self.decoder(fmap1, fmap2, fmap3, fmap4, fmap5)

        if output_size and tuple(segmentation_mask.shape[2:]) != tuple(output_size):
            segmentation_mask = image_resize(segmentation_mask, size=output_size)
        
        return segmentation_mask
    