import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

img = cv2.cvtColor(cv2.imread(f'DJI_0140_002.jpg'), cv2.COLOR_BGR2RGB)
print(img.shape)