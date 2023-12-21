# Crop Mask Generation

## Introduction

This module is used for generating crop mask images and creating a corresponding dataset that includes the ROI coordinates along with their respective harvest dates.

In the initial stage, date files required manual preprocessing to address issues of inconsistency and complexity in their shapes. Subsequently, shapefiles containing the geometries of the crop maps are utilized to define clusters of adjacent fields. Finally, metadata such as crop type, harvest date, country, etc., are incorporated into the output dataset alongside these ROIs.

## Loading Data

The initial raw data for each crop type are located in the `files/Crops/` directory, within their respective subdirectories.

The file `files/Crops/CropsCalendar.xlsx` contains reformatted dates for each crop type in a separate sheet.

The directory `files/Iran Provinces` contains a shapefile outlining the borders of Iran’s provinces.

All these different data are loaded into a specific data structure using the `dataset.py` script to serve as the single source of truth.

Here is an overview of the final dictionary structure:

```py
data = {
    'canola': [
        {
            'gdf': canola,
            'calendar': calendar.parse('Canola'),
            'start_year': '1400',
            'end_year': '1401'
        },
        ...
    ],
    ...
}
```

The final projection for the geodata coordinates is specified in the `PREFERRED_CRS` variable and any shapefile not in this projection will be converted.

## Data Processing

The `main.py` file is the single script that processes the data and creates the dataset.

Important parameters, such as clustering settings and the output path, are listed at the top.

The DBScan clustering algorithm is used to create groups of nearby fields based on their centroid distances, specified in the `CLUSTER_THRESHOLD` variable.

Use 1 as the minimum number of fields in a cluster, specified in the `CLUSTER_MINIMUM_MEMBERS` variable unless you are familiar with your data. Otherwise, it’s possible that a large, useless area may form between fields, or some fields may end up missing from the final dataset.

If the size of an area is less than the value specified by `MINIMUM_PATCH_SIZE`, padding will be added to the mask image to compensate for the deficiency.

Here is the final form of each dataset sample, which can be cherry-picked based on need:

```py
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
```

### Properties:

- cluster: Cluster index.
- uuid: A randomly assigned unique identifier for this sample.
- province: The province where this cluster of fields is located. The value is derived using the `provinces` variable defined in `dataset.py`.
- p_start: Start date of planting.
- p_end: End date of planting.
- h_start: Start date of harvesting.
- h_end: End date of harvesting.
- geometry: Coordinates defining the cluster boundaries.

Here is our sampled version, which will be written to the `Iran_ROI.xlsx` file:

```py
result = {
    'roi': [[
        [patch['geometry'].bounds[0], patch['geometry'].bounds[3]],
        [patch['geometry'].bounds[0], patch['geometry'].bounds[1]],
        [patch['geometry'].bounds[2], patch['geometry'].bounds[1]],
        [patch['geometry'].bounds[2], patch['geometry'].bounds[3]]
    ]],
    'harvest_date': patch['h_start'],
    'main_type': crop,
    'country': 'Iran',
    'uuid': patch['uuid']
}
```
