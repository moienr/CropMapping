import pandas as pd
import geopandas as gpd


PREFERRED_CRS = 'EPSG:3857'

canola = gpd.read_file(
    'files/Crops/Canola/GCP/1400-1401/Canola_GCP_Tr_1400-1401.shp')
cotton = gpd.read_file(
    'files/Crops/Cotton/GCP/1400-1401/Cotton_GCP_Tr_1400-1401.shp')
lentils = gpd.read_file(
    'files/Crops/Lentils/GCP/1400-1401/Lentils_GCP_Tr_1400-1401.shp')
maize = gpd.read_file(
    'files/Crops/Maize/GCP/1400-1401/Maize_GCP_Tr_1400-1401.shp')
onion = gpd.read_file(
    'files/Crops/Onion/GCP/1400-1401/Onion_GCP_Tr_1400-1401.shp')
pea = gpd.read_file('files/Crops/Pea/GCP/1400-1401/Pea_GCP_Tr_1400-1401.shp')
sugarbeet = gpd.read_file(
    'files/Crops/SugerBeet/GCP/1400-1401/Sugerbeet_GCP_Tr_1400-1401.shp')
tomato = gpd.read_file(
    'files/Crops/Tomato/GCP/1400-1401/Tomato_GCP_Tr_1400-1401.shp')

provinces = gpd.read_file('files/Iran Provinces/Iran_Provinces.shp')

calendar = pd.ExcelFile('files/Crops/CropsCalendar.xlsx')

iran_roi = pd.ExcelFile('files/Iran_ROI.xlsx')

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


data = {
    'canola': [
        {
            'gdf': canola,
            'calendar': calendar.parse('Canola'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'cotton': [
        {
            'gdf': cotton,
            'calendar': calendar.parse('Cotton'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'lentils': [
        {
            'gdf': lentils,
            'calendar': calendar.parse('Lentils_Irrigated'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': lentils,
            'calendar': calendar.parse('Lentils_Autumn'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'maize': [
        {
            'gdf': maize,
            'calendar': calendar.parse('Maize'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'onion': [
        {
            'gdf': onion,
            'calendar': calendar.parse('Onion_Spring'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': onion,
            'calendar': calendar.parse('Onion_Summer'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': onion,
            'calendar': calendar.parse('Onion_Autumn'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': onion,
            'calendar': calendar.parse('Onion_Winter'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'pea': [
        {
            'gdf': pea,
            'calendar': calendar.parse('Pea_Irrigated'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': pea,
            'calendar': calendar.parse('Pea_Autumn'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'sugarbeet': [
        {
            'gdf': sugarbeet[sugarbeet['Product'] == 'چغندر قند بهاره'],
            'calendar': calendar.parse('SugarBeet_Spring'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': sugarbeet[sugarbeet['Product'] == 'چغندر قند پاییزه'],
            'calendar': calendar.parse('SugarBeet_Autumn'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ],
    'tomato': [
        {
            'gdf': tomato,
            'calendar': calendar.parse('Tomato_Summer'),
            'start_year': '1400',
            'end_year': '1401'
        },
        {
            'gdf': tomato,
            'calendar': calendar.parse('Tomato_Winter'),
            'start_year': '1400',
            'end_year': '1401'
        }
    ]
}
