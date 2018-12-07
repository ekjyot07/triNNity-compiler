from .helper import *

from .googlenet import GoogleNet
from .vgg import VGG16
from .alexnet import AlexNet
from .caffenet import CaffeNet
from .nin import NiN
from .resnet import ResNet50, ResNet101, ResNet152

# Collection of sample auto-generated models
MODELS = (AlexNet, CaffeNet, GoogleNet, NiN, ResNet50, ResNet101, ResNet152, VGG16)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
# The recommended batch size is based on a Titan X (12GB).

def alexnet_spec(batch_size=500):
    '''Parameters used by AlexNet and its variants.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=227, isotropic=False)


def std_spec(batch_size, isotropic=True):
    '''Parameters commonly used by "post-AlexNet" architectures.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=224, isotropic=isotropic)


MODEL_DATA_SPECS = {
    AlexNet: alexnet_spec(),
    CaffeNet: alexnet_spec(),
    GoogleNet: std_spec(batch_size=200, isotropic=False),
    ResNet50: std_spec(batch_size=25),
    ResNet101: std_spec(batch_size=25),
    ResNet152: std_spec(batch_size=25),
    NiN: std_spec(batch_size=500),
    VGG16: std_spec(batch_size=25)
}


def get_models():
    '''Returns a tuple of sample models.'''
    return MODELS


def get_data_spec(model_instance=None, model_class=None):
    '''Returns the data specifications for the given network.'''
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]
