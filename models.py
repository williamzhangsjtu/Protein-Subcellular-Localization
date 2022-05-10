import torch
import torch.nn as nn
import backbones

class Baseline(nn.Module):
    """
    Baseline model for single-instance classification
    """
    def __init__(self, backbone='densenet121', n_classes=10):
        super(Baseline, self).__init__()
        
        self.backbone = getattr(backbones, backbone)()
        x = torch.randn(1,3,224,224)
        with torch.no_grad():
            n_hidden = self.backbone(x).shape[-1]
        self.classifier = nn.Linear(n_hidden, n_classes)
        

    def forward(self, image):
        hidden = self.backbone(image)
        return self.classifier(hidden)
        

class Reconstructor(nn.Module):
    """
    Multi-task learning of reconstruction
    """
    def __init__(self, backbone='densenet121', n_classes=10):
        super(Reconstructor, self).__init__()
        self.backbone = getattr(backbones, backbone)()
        x = torch.randn(1,3,224,224)
        with torch.no_grad():
            n_hidden = self.backbone(x).shape[-1]
        self.classifier = nn.Linear(n_hidden, n_classes)
        self.decoder = decoder(n_hidden)
        

    def forward(self, image):
        hidden = self.backbone(image)
        if self.training:
            return self.classifier(hidden),\
                    self.decoder(hidden)
        else:
            return self.classifier(hidden)
    def load_param(self, backbone_params, decoder_params=None):
        self.backbone.load_state_dict(backbone_params)

class Mask(nn.Module):
    """ 
    Generating mask
    input: B x C x W x H
    output: B x D; elements in (0, 1)
    """

    def __init__(self, graph_size=224):
        super(Mask, self).__init__()
        down_sampling = nn.ModuleList()
        up_sampling = nn.ModuleList()

        channel = [24, 24, 12, 6, 6, 3]
        for i in range(len(channel) - 1):
            up_sampling.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            up_sampling.append(nn.BatchNorm2d(channel[i + 1]))
            up_sampling.append(nn.ReLU())
        up_sampling.append(nn.Sigmoid())

        channel = channel[::-1]
        for i in range(len(channel) - 1):
            down_sampling.append(nn.Conv2d(channel[i], channel[i + 1], 3, 2, 1))
            down_sampling.append(nn.BatchNorm2d(channel[i + 1]))
            down_sampling.append(nn.ReLU())
        
        self.up_sampling = nn.Sequential(*up_sampling)
        self.down_sampling = nn.Sequential(*down_sampling)

    def forward(self, input):
        down_sample = self.down_sampling(input)
        return self.up_sampling(down_sample)

class MaskModel(nn.Module):
    """
    Multi-task learning of Mask
    """
    def __init__(self, backbone='densenet121', n_classes=10):
        super(MaskModel, self).__init__()
        self.backbone = getattr(backbones, backbone)()
        x = torch.randn(1,3,224,224)
        with torch.no_grad():
            n_hidden = self.backbone(x).shape[-1]
        self.classifier = nn.Linear(n_hidden, n_classes)
        self.mask = Mask()
        

    def forward(self, image):
        mask = self.mask(image)
        mask = (mask >= 0.5).to(torch.float)
        masked_images = mask * image
        hidden = self.backbone(masked_images)
        return self.classifier(hidden)

    def load_param(self, backbone_params, decoder_params=None):
        self.backbone.load_state_dict(backbone_params)


class MultipleInstanceBase(nn.Module):
    """
    Base model for multi-instance learning
    input: B x N x 3 x H x W
    """
    def __init__(self, backbone='densenet121', fusion='CNNFusion', n_classes=10, fusion_args={}, offline=False):
        super(MultipleInstanceBase, self).__init__()
        self.backbone = getattr(backbones, backbone)()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            self.cnn_dim = self.backbone(x).shape[-1]
        self.fusion = eval(fusion)(input_dim=self.cnn_dim, **fusion_args)
        with torch.no_grad():
            x = torch.randn(1,16,self.cnn_dim)
            self.n_hidden = self.fusion(x).shape[-1]
        self.classifier = nn.Linear(self.n_hidden, n_classes)
        self.offline = offline
        if offline:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def load_param(self, backbone_params, decoder_params=None):
        self.backbone.load_state_dict(backbone_params)
        if decoder_params:
            self.decoder.load_state_dict(decoder_params)

class MultipleInstanceReconstructor(MultipleInstanceBase):
    """
    input: B x N x 3 x H x W
    """
    def __init__(self, backbone='densenet121', fusion='CNNFusion', n_classes=10, fusion_args={}, offline=False):
        super(MultipleInstanceReconstructor, self).__init__(
            backbone=backbone, fusion=fusion, n_classes=n_classes, fusion_args=fusion_args, offline=False)
        self.decoder = decoder(self.cnn_dim)

    def forward(self, image):
        (B, N) = image.shape[:2]
        with torch.set_grad_enabled(not self.offline):
            hidden = self.backbone(image.view(-1, *image.shape[2:]))
        fusion_hidden = self.fusion(hidden.view(B, N, -1))
        if self.training:
            return self.classifier(fusion_hidden),\
                self.decoder(hidden)
        else:
            return self.classifier(fusion_hidden)

class MultipleInstanceBaseline(MultipleInstanceBase):
    """
    input: B x N x 3 x H x W
    """
    def __init__(self, backbone='densenet121', fusion='CNNFusion', n_classes=10, fusion_args={}, offline=False):
        super(MultipleInstanceBaseline, self).__init__(
            backbone=backbone, fusion=fusion, n_classes=n_classes, fusion_args=fusion_args, offline=offline)
        
    def forward(self, image):
        (B, N) = image.shape[:2]
        with torch.set_grad_enabled(not self.offline):
            hidden = self.backbone(image.view(-1, *image.shape[2:]))
        fusion_hidden = self.fusion(hidden.view(B, N, -1))
        return self.classifier(fusion_hidden)

class decoder(nn.Module):
    """
    input: B x D
    output: the origin shape
    """

    def __init__(self, inputdim):
        super(decoder, self).__init__()
        self.proj = nn.Linear(inputdim, 3 * 8 * 7 * 7)
        cnn_module = nn.ModuleList()
        channel = [24, 24, 12, 6, 6, 3]
        for i in range(len(channel) - 1):
            cnn_module.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            cnn_module.append(nn.BatchNorm2d(channel[i + 1]))
            cnn_module.append(nn.ReLU())
        cnn_module[-1] = nn.Tanh()
        self.cnn_module = nn.Sequential(*cnn_module)


    def forward(self, input):
        proj = self.proj(input)
        proj = proj.view(input.shape[0], 3 * 8, 7, 7)

        return self.cnn_module(proj)#.squeeze(1)

class CNNFusion(nn.Module):
    def __init__(self, **kwargs):
        super(CNNFusion, self).__init__()
        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.Dropout(p=0.2))
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 5), padding=1, stride=(2, 3)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.Dropout(p=0.2))
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 5), padding=1, stride=(2, 3)), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.Dropout(p=0.2))
            

        self.model = nn.Sequential(
            cnn_block1, cnn_block2, cnn_block3,
            nn.AdaptiveAvgPool2d(1))
    
    def forward(self, x):
        return self.model(x.unsqueeze(1)).flatten(1, 3)


class CNNBigFusion(nn.Module):
    def __init__(self, **kwargs):
        super(CNNBigFusion, self).__init__()
        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.Dropout(p=0.2))
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.Dropout(p=0.2))
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 3), padding=1, stride=(2, 2)), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.Dropout(p=0.2))
        cnn_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=(2, 2)), 
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.Dropout(p=0.2))
        cnn_block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, stride=(2, 2)), 
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=(1, 2)), 
            nn.Dropout(p=0.2))

        self.model = nn.Sequential(
            cnn_block1, cnn_block2, cnn_block3,
            cnn_block4, cnn_block5,
            nn.AdaptiveAvgPool2d(1))
    
    def forward(self, x):
        return self.model(x.unsqueeze(1)).flatten(1, 3)

        
class SimpleAttn(nn.Module):
    def __init__(self, dim):
        super(SimpleAttn, self).__init__()
        self.Linear = nn.Linear(dim, 1)
        self.Softmax = nn.Softmax(dim=1)
        nn.init.normal_(self.Linear.weight, 0, 0.1)
        nn.init.constant_(self.Linear.bias, 0.0)

    def forward(self, input, input_num=None):
        if (input_num is not None):
            idxs = torch.arange(input.shape[1]).repeat(input.shape[0]).view(input.shape[:2])
            masks = idxs.cpu() < input_num.cpu().view(-1, 1)
            masks = masks.to(torch.float).to(input.device)
            input = input * masks.unsqueeze(-1)
        
        alpha = self.Softmax(self.Linear(input))
        output = (alpha * input).sum(1)
        return output # B x D


class MultiHeadFusion(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(MultiHeadFusion, self).__init__()
        num_heads = kwargs.get('num_heads', 4)
        dropout = kwargs.get('dropout', 0.2)
        self.multihead_attn = nn.MultiheadAttention(
            input_dim, num_heads=num_heads, dropout=dropout)
        self.fusion_attn = SimpleAttn(input_dim)
    
    def forward(self, x):
        x = x.transpose(0, 1)
        multihead_attn_out, _ = self.multihead_attn(x, x, x)

        return self.fusion_attn(multihead_attn_out.transpose(0, 1))

class MeanFusion(nn.Module):
    def __init__(self,  **kwargs):
        super(MeanFusion, self).__init__()
    
    def forward(self, x):
        return x.mean(dim=1)

class TransformerFusion(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(TransformerFusion, self).__init__()
        n_heads = kwargs.get('num_heads', 2)
        dropout = kwargs.get('dropout', 0.2)
        n_layers = kwargs.get('n_layers', 2)
        layer = nn.TransformerEncoderLayer(input_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fusion_attn = SimpleAttn(input_dim)
    
    def forward(self, x):
        x = x.transpose(0, 1)
        transformer_emb = self.transformer(x).transpose(0, 1)

        return self.fusion_attn(transformer_emb)
