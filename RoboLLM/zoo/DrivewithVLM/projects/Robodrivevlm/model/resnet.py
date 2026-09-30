from mmdet.models.backbones import ResNet

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ResNet(ResNet):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(depth=50)
