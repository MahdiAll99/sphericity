from copy import deepcopy
from typing import List, Tuple, Union
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
from pathlib import Path




test = nib.load('./test_data/ct_mask.nii.gz')
print(test)

# visualize mask
import matplotlib.pyplot as plt
plt.imshow(test.get_fdata()[:,:,0], cmap='gray')
plt.show()
