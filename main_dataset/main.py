import os
import dataset
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import box, Polygon
from khayyam import JalaliDate

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
                'province': province,
                'p_start': JalaliDate(dates[0][0], dates[0][1], dates[0][2]).todate().strftime('%Y-%m-%d'),
                'p_end': JalaliDate(dates[1][0], dates[1][1], dates[1][2]).todate().strftime('%Y-%m-%d'),
                'h_start': JalaliDate(dates[2][0], dates[2][1], dates[2][2]).todate().strftime('%Y-%m-%d'),
                'h_end': JalaliDate(dates[3][0], dates[3][1], dates[3][2]).todate().strftime('%Y-%m-%d'),
                'geometry': patch_bounding_box
            }

            patches.loc[len(patches)] = patch

        os.makedirs(os.path.join(OUTPUT_DIRECTORY, crop,
                    OUTPUT_SHAPEFILE_DIRECTORY), exist_ok=True)

        # Output
        output_path_shapefile = os.path.join(
            OUTPUT_DIRECTORY, crop, OUTPUT_SHAPEFILE_DIRECTORY, f"{crop}.shp")
        patches.to_file(output_path_shapefile)
