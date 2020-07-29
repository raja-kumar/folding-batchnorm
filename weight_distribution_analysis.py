from bn_folder import bn_folding_model
from MobileNetV2 import mobilenet_v2
import pandas as pd
import numpy as np


net = mobilenet_v2('./mobilenetv2_1.0-f2a8633.pth.tar')

folded_net = bn_folding_model(net)

weight_temp = new_net.features[1].conv[0].weight

imgs = weight_temp.detach().numpy()[:, ::-1, :, :]

x = imgs.flatten()
x = x.reshape(-1,32)
#print(x)
#print(np.shape(x))
df = pd.DataFrame(x)
df.plot.box(grid = True)
