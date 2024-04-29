%pip install Pillow

import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# 이미지를 불러옵니다.
image = Image.open('/content/drive/MyDrive/AI_proj/Project/id_photo_crops/0/0.jpg')

# ColorJitter를 설정합니다.
# brightness=0.2는 밝기를 최대 20%까지 변화시킬 수 있음을 의미합니다.
# contrast=0.3는 대비를 최대 30%까지 변화시킬 수 있음을 의미합니다.
# saturation=0.2는 채도를 최대 20%까지 변화시킬 수 있음을 의미합니다.
# hue=0.1는 색조를 최대 10%까지 변화시킬 수 있음을 의미합니다.
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1)

# 이미지에 변환을 적용합니다.
jittered_image = color_jitter(image)

# 변형된 이미지를 저장하거나 보여줍니다.
jittered_image.save('/content/drive/MyDrive/AI_proj/Project/id_photo_crops/0/0_jittered.jpg')
jittered_image.show()

