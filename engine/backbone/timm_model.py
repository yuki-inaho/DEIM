"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""


import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from .utils import IntermediateLayerGetter
from ..core import register


@register()
class TimmModel(torch.nn.Module):
    def __init__(self, name, return_layers, pretrained=False, exportable=True, features_only=True, **kwargs) -> None:
        super().__init__()

        import timm

        model = timm.create_model(
            name, pretrained=pretrained, exportable=exportable, features_only=features_only, **kwargs
        )
        # nodes, _ = get_graph_node_names(model)
        # print(nodes)

        # features = {'': ''}
        # model = create_feature_extractor(model, return_nodes=features)

        assert set(return_layers).issubset(
            model.feature_info.module_name()
        ), f"return_layers should be a subset of {model.feature_info.module_name()}"

        # Create a mapping to replace layer names written in feature_info with layer names in PyTorch
        feature_to_child_map = {}
        for m_name in model.feature_info.module_name():
            # Example: 'stages.0' -> 'stages_0'
            #          'stages.1' -> 'stages_1' ...
            mapped_name = m_name.replace(".", "_")
            feature_to_child_map[m_name] = mapped_name

        # Replace 'stages.0', 'stages.1', etc. in the received return_layers
        # with 'stages_0', 'stages_1', etc. and pass them to IntermediateLayerGetter
        new_return_layers = []
        for layer in return_layers:
            new_return_layers.append(feature_to_child_map[layer])

        self.model = IntermediateLayerGetter(model, return_layers=new_return_layers)

        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        # outputs = [outputs[i] for i in self.return_idx]
        return outputs


if __name__ == "__main__":
    model = TimmModel(name="resnet34", return_layers=["layer2", "layer3"])
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)

    for output in outputs:
        print(output.shape)

    """
    model:
        type: TimmModel
        name: resnet34
        return_layers: ['layer2', 'layer4']
    """
