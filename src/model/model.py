import segmentation_models_pytorch as smp
from torch import nn


class Unet(nn.Module):
    def __init__(self):
        """
               U-Net model using a ResNet34 encoder for image segmentation.

               The weights of the encoder are initialized with pre-trained ImageNet weights, and the parameters of
               the layers before the last residual block in the encoder are frozen.

               The model expects input result with three channels and outputs segmentation masks with one channel.

               """
        super(Unet, self).__init__()
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
        for name, param in self.named_parameters():
            if name == 'unet.encoder.layer4.2.bn2.bias':
                break
            param.requires_grad = False

    def forward(self, x):
        """
                Forward pass of the U-Net model.

                Args:
                    x (torch.Tensor): Input tensor representing an image.

                Returns:
                    torch.Tensor: Output tensor representing the segmentation mask.
                """
        return self.unet(x)