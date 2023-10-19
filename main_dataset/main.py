import os
import dataset
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import box

# Parameters
TARGET_CRS = 'EPSG:3857'

# Clustering
# Maximum distance between the center points of polygons that will be in the same cluster
CLUSTER_THRESHOLD = 2000

# Minimum number of polygons necessary to form a cluster
CLUSTER_MINIMUM_MEMBERS = 1

# Output
OUTPUT_DIRECTORY = 'output'
OUTPUT_SHAPEFILE_DIRECTORY = 'shapefile'


dbscan = DBSCAN(eps=CLUSTER_THRESHOLD, min_samples=CLUSTER_MINIMUM_MEMBERS)
for crop, data in dataset.data.items():
    for sample in data:

        patches = gpd.GeoDataFrame(
            {
                'cluster': [],
                'geometry': []
            },
            crs=TARGET_CRS
        )

        gdf = sample['gdf'].copy()

        centroids = gdf.geometry.centroid.apply(
            lambda point: (point.x, point.y)).values.tolist()
        labels = dbscan.fit_predict(centroids)
        gdf['cluster'] = labels

        for cluster in gdf['cluster'].unique():
            patch_bounds = gdf[gdf['cluster'] == cluster].total_bounds
            patch_bounding_box = box(*patch_bounds)

            patch = {
                'cluster': cluster,
                'geometry': patch_bounding_box
            }

            patches.loc[len(patches)] = patch

        os.makedirs(os.path.join(OUTPUT_DIRECTORY, crop,
                    OUTPUT_SHAPEFILE_DIRECTORY), exist_ok=True)

        # Output
        output_path_shapefile = os.path.join(
            OUTPUT_DIRECTORY, crop, OUTPUT_SHAPEFILE_DIRECTORY, f"{crop}.shp")
        patches.to_file(output_path_shapefile)
