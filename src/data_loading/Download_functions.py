import os
import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Polygon


def import_boundary(boundaryPath, buffer=0.05):
    boundary = gpd.read_file(boundaryPath)
    boundary = boundary.geometry
    if boundary.crs != "EPSG:4326":
        print("Converting boundary CRS")
        boundary = boundary.to_crs("EPSG:4326")
    if len(boundary) > 1:
        boundary = boundary.unary_union
    buff = boundary.buffer(buffer)
    buff = gpd.GeoSeries(Polygon(buff[0].exterior), crs="EPSG:4326")
    buff = gpd.GeoDataFrame(geometry=buff)
    filename = os.path.splitext(boundaryPath)[0] + "_buffer" + str(buffer) + ".shp"
    buff.to_file(filename)
    return buff


def get_tiles(shp, gridPath):
    grid = gpd.read_file(gridPath)
    tiles = gpd.sjoin(grid, shp, op="intersects")
    return tiles


"""
def extract_corners(tiles):
    for index, row in tiles.iterrows():
        coords = np.array(row["geometry"].exterior.coords)
        lat, lon = np.max(coords[:, 1]), np.min(coords[:, 0])
        if "lats" in locals():
            lats = np.append(lats, lat)
            lons = np.append(lons, lon)
        else:
            lats = lat
            lons = lon
    corners = np.column_stack([lats, lons])
    del lats
    del lons
    return corners
"""


def extract_corners(tiles):
    lats, lons = [], []
    for index, row in tiles.iterrows():
        coords = np.array(row["geometry"].exterior.coords)
        lat, lon = np.max(coords[:, 1]), np.min(coords[:, 0])
        lats = np.append(lats, lat)
        lons = np.append(lons, lon)
    corners = np.column_stack([lats, lons])
    del lats
    del lons
    return corners


def create_end(tiles):
    corners = extract_corners(tiles)
    ends = ["A"] * len(corners)
    for i in range(len(corners)):
        if corners[i, 0] < 0:
            NS = str(np.int(np.abs(corners[i, 0]))) + "S"
        else:
            NS = str(np.int(np.abs(corners[i, 0]))) + "N"
        if len(NS) == 2:
            NS = "0" + NS
        if corners[i, 1] < 0:
            EW = str(np.int(np.abs(corners[i, 1]))) + "W"
        else:
            EW = str(np.int(np.abs(corners[i, 1]))) + "E"
        while len(EW) < 4:
            EW = "0" + EW
        ends[i] = "_" + NS + "_" + EW + ".tif"
    return ends


def create_filenames(tiles, year, type):
    ends = create_end(tiles)
    names = ["A"] * len(tiles)
    for i in range(len(ends)):
        if year < 2015:
            names[i] = "Hansen_GFC" + str(year) + "_" + str(type) + ends[i]
        if year >= 2015:
            names[i] = (
                "Hansen_GFC-"
                + str(year)
                + "-v1."
                + str(year - 2012)
                + "_"
                + str(type)
                + ends[i]
            )
    return names


def create_links(tiles, year, type):
    root = "https://storage.googleapis.com/earthenginepartners-hansen/"
    names = create_filenames(tiles, year, type)
    links = ["A"] * len(tiles)
    for i in range(len(names)):
        if year < 2015:
            links[i] = root + "GFC" + str(year) + "/" + names[i]
        if year >= 2015:
            links[i] = (
                root + "GFC-" + str(year) + "-v1." + str(year - 2012) + "/" + names[i]
            )
    return links


"""
def create_links(tiles, year, type):
    root = "https://storage.googleapis.com/earthenginepartners-hansen/"
    ends = create_end(tiles)
    links = ["A"] * len(tiles)
    for i in range(len(ends)):
        if year < 2015:
            links[i] = (
                root
                + "GFC"
                + str(year)
                + "/Hansen_GFC"
                + str(year)
                + "_"
                + str(type)
                + ends[i]
            )
        if year >= 2015:
            links[i] = (
                root
                + "GFC-"
                + str(year)
                + "-v1."
                + str(year - 2012)
                + "/Hansen_GFC-"
                + str(year)
                + "-v1."
                + str(year - 2012)
                + "_"
                + str(type)
                + ends[i]
            )
    return links
"""

"""
def download_data(shp, dwnldPath, years, types, gridPath):
    # tiles = gpd.sjoin(shp, grid)
    tiles = get_tiles(shp, gridPath)
    for year in years:
        for type in types:
            links = create_links(tiles, year, type)
            for link in links:
                r = requests.get(link, stream=True)
                size = int(r.headers.get("content-length", None))
                filedest = dwnldPath + "/" + os.path.basename(link)
                if os.path.exists(filedest):
                    if os.path.getsize(filedest) == size:
                        print("Already downloaded - skipping " + type + " " + str(year))
                        continue
                else:
                    with open(filedest, "wb") as file:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                file.write(chunk)
                    if os.path.getsize(filedest) == size:
                        print("Download complete " + type + " " + str(year))
                    else:
                        print(
                            "Download is of unexpected size " + type + " " + str(year)
                        )
"""


def download_data(shp, dwnldPath, years, types_static, types_dynamic, gridPath):
    # tiles = gpd.sjoin(shp, grid)
    if not os.path.exists(dwnldPath):
        os.makedirs(dwnldPath)
    years.sort()
    tiles = get_tiles(shp, gridPath)
    for year in years:
        if year == years[-1]:
            types = types_static + types_dynamic
        else:
            types = types_dynamic
        for type in types:
            links = create_links(tiles, year, type)
            for link in links:
                r = requests.get(link, stream=True)
                size = int(r.headers.get("content-length", None))
                filedest = dwnldPath + "/" + os.path.basename(link)
                if os.path.exists(filedest):
                    if os.path.getsize(filedest) == size:
                        print("Already downloaded - skipping " + type + " " + str(year))
                        continue
                    else:
                        print("Downloaded file is not of correct size")
                        print("Deleting file and redownloading")
                        os.remove(filedest)
                        with open(filedest, "wb") as file:
                            for chunk in r.iter_content(chunk_size=1024):
                                if chunk:
                                    file.write(chunk)
                        if os.path.getsize(filedest) == size:
                            print("Download complete " + type + " " + str(year))
                        else:
                            print("DOWNLOAD IS STILL OF UNEXPECTED SIZE " + link)
                            print("RETRY DOWNLOAD SCRIPT")
                            raise Exception("Download of file not completing properly")
                else:
                    with open(filedest, "wb") as file:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                file.write(chunk)
                    if os.path.getsize(filedest) == size:
                        print("Download complete " + type + " " + str(year))
                    else:
                        print(
                            "DOWNLOAD IS OF UNEXPECTED SIZE " + type + " " + str(year)
                        )
                        print("RETRY DOWNLOAD SCRIPT")
                        break

