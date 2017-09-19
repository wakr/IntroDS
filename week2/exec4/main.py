import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import matplotlib.pyplot as plt

world = gpd.read_file("week2/exec4/world_m/world_m.shp")
cities = gpd.read_file("week2/exec4/cities/cities.shp")

world.plot(figsize=(40, 40), facecolor="gray")
cities.plot(figsize=(40, 40), facecolor="gray")

cities_proj = cities.copy()
cities_proj['geometry'] = cities_proj['geometry'].to_crs(epsg=3395)


world_proj = world.copy()
world_proj['geometry'] = world_proj['geometry'].to_crs(epsg=3395)


world_proj.plot(ax=cities_proj.plot(figsize=(50, 50), marker='.', color='red', markersize = 100), facecolor="gray")

