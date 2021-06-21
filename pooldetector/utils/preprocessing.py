from pathlib import Path
import shutil
import os
import random
from pathlib import Path
from IPython.core.debugger import set_trace

import geopandas as gpd
import pandas as pn
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.plot import show
from rasterio.plot import reshape_as_image

import matplotlib.pyplot as plt
from matplotlib import patches

from tqdm.notebook import tqdm
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

def tile_big_image(
    img_src: str,
    bbox_src: str,
    overlap: float,
    out_img_dir: str = "./images/",
    out_labels_dir: str = "./labels/",
):
    pass

