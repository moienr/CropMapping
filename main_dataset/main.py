import os

import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box, Polygon
from sklearn.cluster import DBSCAN
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from khayyam import JalaliDate
import uuid

import dataset


# Parameters
TARGET_CRS = 'EPSG:3857'

# Clustering
# Maximum distance between the center points of polygons that will be in the same cluster
CLUSTER_THRESHOLD = 2000

# Minimum number of polygons necessary to form a cluster
CLUSTER_MINIMUM_MEMBERS = 1

# Mask properties
MASK_RESOLUTION = 10  # Meter per pixel
MINIMUM_PATCH_SIZE = 64  # Pixels

# Output
OUTPUT_DIRECTORY = 'output'
OUTPUT_SHAPEFILE_DIRECTORY = 'shapefile'
OUTPUT_MASKS_DIRECTORY = 'masks'


def adjust_box_minimum_size(bounds, minimum_size=64):
    minx, miny, maxx, maxy = bounds

    width = maxx - minx
    height = maxy - miny

    if width < minimum_size:
        difference = (minimum_size - width) / 2
        minx -= difference
        maxx += difference

    if height < minimum_size:
        difference = (minimum_size - height) / 2
        miny -= difference
        maxy += difference

    return (minx, miny, maxx, maxy)


def get_year_transition_index(months):
    for i in range(1, len(months)):
        if months[i] < months[i - 1]:
            return i
    return -1


dbscan = DBSCAN(eps=CLUSTER_THRESHOLD, min_samples=CLUSTER_MINIMUM_MEMBERS)

for crop, data in dataset.data.items():
    for sample in data:

        patches = gpd.GeoDataFrame(
            {
                'cluster': [],
                'uuid': [],
                'province': [],
                'p_start': [],
                'p_end': [],
                'h_start': [],
                'h_end': [],
                'geometry': []
            },
            crs=TARGET_CRS
        )

        gdf = sample['gdf'].copy()
        calendar = sample['calendar'].copy()

        centroids = gdf.geometry.centroid.apply(
            lambda point: (point.x, point.y)).values.tolist()
        labels = dbscan.fit_predict(centroids)
        gdf['cluster'] = labels

        for cluster in gdf['cluster'].unique():
            patch_bounds = gdf[gdf['cluster'] == cluster].total_bounds
            patch_bounding_box = box(*patch_bounds)
            patch_adjusted_bounding_box = box(
                *adjust_box_minimum_size(patch_bounds, MINIMUM_PATCH_SIZE * MASK_RESOLUTION))

            province = [p['province'] for _, p in dataset.provinces.iterrows(
            ) if p.geometry.contains(patch_bounding_box)]
            if len(province) > 0:
                province = province[0]
            else:
                intersections = [p.geometry.intersection(patch_bounding_box) if p.geometry.intersects(
                    patch_bounding_box) else Polygon([]) for _, p in dataset.provinces.iterrows()]
                areas = [
                    intersection.area if intersection else 0 for intersection in intersections]
                province = dataset.provinces.iloc[areas.index(
                    max(areas))]['province']

            dates = calendar[calendar['Province'] == province]
            if not dates.empty:
                dates_year = []

                dates_month = [
                    dates['Planting Start Date'].iloc[0].split('-')[0],
                    dates['Planting End Date'].iloc[0].split('-')[0],
                    dates['Harvest Start Date'].iloc[0].split('-')[0],
                    dates['Harvest End Date'].iloc[0].split('-')[0]
                ]

                dates_day = [
                    dates['Planting Start Date'].iloc[0].split('-')[1],
                    dates['Planting End Date'].iloc[0].split('-')[1],
                    dates['Harvest Start Date'].iloc[0].split('-')[1],
                    dates['Harvest End Date'].iloc[0].split('-')[1]
                ]

                if any(i == '' for i in dates_month) or any(i == '' for i in dates_day):
                    continue

                year_transition_index = get_year_transition_index(
                    [int(i) for i in dates_month])

                if year_transition_index == -1:
                    # One-year agriculture format
                    dates_year = [sample['start_year']
                                  for i in range(len(dates_month))]
                else:
                    # Two-year agriculture format
                    dates_year = [sample['start_year'] if i < year_transition_index else sample['end_year']
                                  for i in range(len(dates_month))]

                dates = [[int(i) for i in date]
                         for date in zip(dates_year, dates_month, dates_day)]
            else:
                # TODO: For future processing, store clusters that have no date
                continue

            patch = {
                'cluster': cluster,
                'uuid': uuid.uuid4().hex,
                'province': province,
                'p_start': JalaliDate(dates[0][0], dates[0][1], dates[0][2]).todate().strftime('%Y-%m-%d'),
                'p_end': JalaliDate(dates[1][0], dates[1][1], dates[1][2]).todate().strftime('%Y-%m-%d'),
                'h_start': JalaliDate(dates[2][0], dates[2][1], dates[2][2]).todate().strftime('%Y-%m-%d'),
                'h_end': JalaliDate(dates[3][0], dates[3][1], dates[3][2]).todate().strftime('%Y-%m-%d'),
                'geometry': patch_adjusted_bounding_box
            }
            patches.loc[len(patches)] = patch

        # directories
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, crop,
                    OUTPUT_SHAPEFILE_DIRECTORY), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, crop,
                    OUTPUT_MASKS_DIRECTORY), exist_ok=True)

        # Output
        output_path_shapefile = os.path.join(
            OUTPUT_DIRECTORY, crop, OUTPUT_SHAPEFILE_DIRECTORY, f"{crop}.shp")
        patches.to_file(output_path_shapefile)

        for _, row in patches.iterrows():
            farms = gdf[gdf['cluster'] == row['cluster']]

            minx, miny, maxx, maxy = row.geometry.bounds
            width = int((maxx - minx) / MASK_RESOLUTION)
            height = int((maxy - miny) / MASK_RESOLUTION)

            image = np.zeros((height, width), dtype=np.uint8)
            transform = from_origin(
                minx, maxy, MASK_RESOLUTION, MASK_RESOLUTION)
            mask = geometry_mask(
                farms.geometry, transform=transform, invert=True, out_shape=(height, width))
            image[mask] = 255

            output_path_mask = os.path.join(
                OUTPUT_DIRECTORY, crop, OUTPUT_MASKS_DIRECTORY, f"{row['uuid']}.png")
            with Image.fromarray(image) as image:
                image.save(output_path_mask)
