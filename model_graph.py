import torch
from torchviz import make_dot
from model import LightSegNet_V3

model = LightSegNet_V3()

x = torch.randn(1,3,300,300)
y = model(x)

make_dot(y, params=dict(model.named_parameters())).render("model_graph", format="png")