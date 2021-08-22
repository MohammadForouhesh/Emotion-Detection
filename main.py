import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))