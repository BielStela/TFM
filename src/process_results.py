import argparse

import geopandas as gpd
from shapely.geometry import box


def remove_sea_polygons(df: gpd.GeoDataFrame, coastline_file: str = "../data/coastline.shp") -> gpd.GeoDataFrame:
    """Removes polygons that fall outside of defined coastline"""
    print("Removing oversea polygons...")
    coastline = gpd.read_file(coastline_file)
    within_land = df.within(coastline.loc[0, "geometry"])  # slow
    return df.loc[within_land]


def melt_overlapping(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # merge overlapping polygons originated from the windowed images
    # Union the overlapping polygons
    print("Melting overlapping polygons...")
    merged = gpd.GeoDataFrame(geometry=list(df.unary_union), crs=df.crs)
    merged['polyID'] = merged.index  # keep poly index
    # create centroids to join later and keep original df data
    points = df.copy()
    points['geometry'] = points['geometry'].centroid
    join_df = gpd.sjoin(merged, points, op='contains')
    join_df = join_df.dissolve(by='polyID', aggfunc='mean')  # mean of confidences
    join_df = join_df.drop(["index_right"], axis=1)
    res_bbox = join_df.copy()
    res_bbox["geometry"] = res_bbox.geometry.apply(lambda x: box(*x.bounds))
    res_bbox = res_bbox.set_crs(df.crs)
    return res_bbox


def clean_results(results: str, coastline: str, outfile: str):
    # load results
    df = gpd.read_file(results)
    print(f"Loaded geodataframe with {df.shape[0]} polygons")
    df = remove_sea_polygons(df, coastline_file=coastline)
    df = melt_overlapping(df)
    print(f"Writing {df.shape[0]} cleaned polygons to {outfile}")
    df.to_file(outfile, driver='GeoJSON')
    print("Done!")


if __name__ == "__main__":
    description = ("Cleans model results by removing sea polygons and disolving overlapping "
                   "polygons originated from rolling window")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("results", help="Result shapefile or geojson returned by a model prediction")
    parser.add_argument("outfile", help="file to store the cleaned data")
    parser.add_argument("coastline", help="file to store the cleaned data")
    args = parser.parse_args()

    clean_results(args.results, args.coastline, args.outfile)
