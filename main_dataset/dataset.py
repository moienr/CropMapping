import pandas as pd
import geopandas as gpd


PREFERRED_CRS = 'EPSG:3857'

canola = gpd.read_file(
    'dataset/Crops/Canola/GCP/1400-1401/Canola_GCP_Tr_1400-1401.shp')
cotton = gpd.read_file(
    'dataset/Crops/Cotton/GCP/1400-1401/Cotton_GCP_Tr_1400-1401.shp')
lentils = gpd.read_file(
    'dataset/Crops/Lentils/GCP/1400-1401/Lentils_GCP_Tr_1400-1401.shp')
maize = gpd.read_file(
    'dataset/Crops/Maize/GCP/1400-1401/Maize_GCP_Tr_1400-1401.shp')
onion = gpd.read_file(
    'dataset/Crops/Onion/GCP/1400-1401/Onion_GCP_Tr_1400-1401.shp')
pea = gpd.read_file('dataset/Crops/Pea/GCP/1400-1401/Pea_GCP_Tr_1400-1401.shp')
sugarbeet = gpd.read_file(
    'dataset/Crops/SugerBeet/GCP/1400-1401/Sugerbeet_GCP_Tr_1400-1401.shp')
tomato = gpd.read_file(
    'dataset/Crops/Tomato/GCP/1400-1401/Tomato_GCP_Tr_1400-1401.shp')


if canola.crs != PREFERRED_CRS:
    canola = canola.to_crs(PREFERRED_CRS)

if cotton.crs != PREFERRED_CRS:
    cotton = cotton.to_crs(PREFERRED_CRS)

if lentils.crs != PREFERRED_CRS:
    lentils = lentils.to_crs(PREFERRED_CRS)

if maize.crs != PREFERRED_CRS:
    maize = maize.to_crs(PREFERRED_CRS)

if onion.crs != PREFERRED_CRS:
    onion = onion.to_crs(PREFERRED_CRS)

if pea.crs != PREFERRED_CRS:
    pea = pea.to_crs(PREFERRED_CRS)

if sugarbeet.crs != PREFERRED_CRS:
    sugarbeet = sugarbeet.to_crs(PREFERRED_CRS)

if tomato.crs != PREFERRED_CRS:
    tomato = tomato.to_crs(PREFERRED_CRS)
