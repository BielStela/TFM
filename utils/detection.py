import argparse
import sys
import time
from itertools import product
from pathlib import Path, PosixPath

import numpy as np
import rasterio
from pytorchyolo import detect, models
from rasterio.plot import reshape_as_image
from shapely.geometry import Polygon
from tqdm import tqdm
import geopandas as gpd

from utils.process_results import remove_sea_polygons, melt_overlapping


def overlapping_windows(src, overlap, width, height):
    """"width & height not including overlap i.e requesting a 256x256 window with
        1px overlap will return a 258x258 window (for non edge windows)"""
    offsets = product(range(0, src.meta['width'], width), range(0, src.meta['height'], height))
    for col_off, row_off in offsets:
        window = rasterio.windows.Window(
            col_off=col_off - overlap,
            row_off=row_off - overlap,
            width=width + overlap * 2,
            height=height + overlap * 2)
        yield window


def detect_image(img: str, model_) -> gpd.GeoDataFrame:
    with rasterio.open(img) as src:
        data = {"geometry": [], "confidence": []}
        count = 0
        pbar = tqdm(overlapping_windows(src, 30, 452, 452))
        for window in pbar:
            pbar.set_postfix({"Pools found": count})
            img_window = reshape_as_image(src.read(window=window))
            detected_bbox = detect.detect_image(model_, img_window, conf_thres=0.15, nms_thres=0.15)
            if detected_bbox.size > 0:
                count += detected_bbox.shape[0]
                # get the window transform matrix
                transform = rasterio.windows.transform(window, src.transform)
                for x1, y1, x2, y2, conf, _ in detected_bbox:
                    # transform from pixels coordinates to geografic coordinates
                    # using the affine transformation matrix
                    geo_x1, geo_y1 = transform * np.array([x1, y1])
                    geo_x2, geo_y2 = transform * np.array([x2, y2])
                    coords = ((geo_x1, geo_y1), (geo_x2, geo_y1), (geo_x2, geo_y2), (geo_x1, geo_y2))
                    data["geometry"].append(Polygon(coords))
                    data["confidence"].append(conf)
    return gpd.GeoDataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", dest="dir", type=str, required=True, help="Directory containing the images to use in the detection")
    parser.add_argument("-o", "--out", dest="out", type=str, required=True, help="Directory to save the results")
    parser.add_argument("-m", "--model", dest="model", type=str, required=True, help="Path to model weights")
    parser.add_argument("-cfg", "--config", dest="config", type=str, required=True, help="Path to model config")
    parser.add_argument("-cst", "--coastline", dest="coastline", type=str, required=True, help="Path to coastline")
    args = parser.parse_args()

    images = list(Path(args.dir).glob("*.tif"))
    if len(images) == 0:
        print("Directory contains no images.")
        sys.exit(0)
    if input(f"You are about to run the detection on {len(images)} images. Continue?") not in ["y", "yes"]:
        sys.exit(0)

    out_path = Path(args.out).resolve()
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = models.load_model(args.config, args.model)

    for img in images:
        fname = img.name.split(".")[0]
        print("Working on image ", str(img.name))
        res = detect_image(img.resolve(), model)
        print(f"Detected {res.shape[0]} objects")
        res = remove_sea_polygons(res, args.coastline)
        res = melt_overlapping(res)
        print(f"Saving cleaned results with {res.shape[0]} pool detections")
        out_file = out_path / (fname + ".geojson")
        res.to_file(out_file, driver='GeoJSON')






