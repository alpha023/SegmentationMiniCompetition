from torchinfo import summary
from model import LightSegNet_V3

model = LightSegNet_V3()

summary(model, input_size=(1,3,300,300))