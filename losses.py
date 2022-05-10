import torch.nn as nn

BCEWithLogitsLoss = nn.BCEWithLogitsLoss

class ReconstructionLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super(ReconstructionLoss, self).__init__()
        self.recon_loss = nn.MSELoss()
        self.classify_loss = BCEWithLogitsLoss()
        self.lambd = lambd
    

    def forward(self, output, label, img):
        if isinstance(output, tuple):
            pred, recon_img = output
            if img.ndim == 5:
                img = img.view(-1, *img.shape[2:])
            return self.lambd * self.classify_loss(pred, label)\
                + (1 - self.lambd) * self.recon_loss(recon_img, img)
        else:
            return self.classify_loss(output, label)