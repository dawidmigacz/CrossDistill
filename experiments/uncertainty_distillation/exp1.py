from lib.models.centernet3d_distill import MonoDistill
from torchsummary import summary


model = MonoDistill()

summary(model)