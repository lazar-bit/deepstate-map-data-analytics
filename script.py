#!/usr/bin/env python3

import os
import sys
import logging
import time
from datetime import datetime
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import wkt
from shapely.geometry import JOIN_STYLE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_URL = "https://deepstatemap.live/api/history/last"
OUTPUT_DIR = "data"
OUTPUT_FILENAME = f"deepstatemap_data_{datetime.now().strftime('%Y%m%d')}.geojson"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

CSV_PATH = os.path.join(OUTPUT_DIR, "aggregated_deepstatemap.csv")


def make_api_request():
    """Make a request to the API and return the JSON response."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All API request attempts failed.")
                sys.exit(1)


def process_data(data):
    """Process the API response data."""
    geo_list = []
    for f in data['map']['features']:
        geom = f['geometry']
        name = f['properties']['name']
        new_feature = {
            "name": name,
            "geometry": wkt.loads(wkt.dumps(shape(geom), output_dimension=2))
        }
        geo_list.append(new_feature)

    # Split the name by '///' and take the second part (part=1)
    def extract_first_part(name, part=1):
        first_part = name.split('///')[part].strip()
        return first_part

    for item in geo_list:
        item['name'] = extract_first_part(item['name'], part=1)

    return geo_list


def create_geodataframe(geo_list):
    """Create a GeoDataFrame from the processed data."""
    raw_gdf = gpd.GeoDataFrame(geo_list).set_crs(4326)

    mask = raw_gdf.geometry.apply(lambda x: isinstance(x, Polygon))
    polygon_gdf = raw_gdf[mask]

    filtered_gdf = polygon_gdf[polygon_gdf['name'].isin(['CADR and CALR', 'Occupied', 'Occupied Crimea'])].reset_index()

    merged_gdf = gpd.GeoSeries(filtered_gdf.geometry.unary_union, crs=4326)

    # Applying buffer to remove union artifacts
    eps = 0.000009

    deartifacted_gdf = (
        merged_gdf
        .buffer(eps, 1, join_style=JOIN_STYLE.mitre)
        .buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
    )

    return deartifacted_gdf


# --- CSV aggregation related functions ---

def extract_date_from_filename(filename):
    basename = os.path.basename(filename)
    date_str = basename.replace("deepstatemap_data_", "").replace(".geojson", "")
    try:
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        return None


def process_geojson(filepath):
    date = extract_date_from_filename(filepath)
    if date is None:
        return pd.DataFrame()

    gdf = gpd.read_file(filepath)
    processed_rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue

        polygons = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        for polygon in polygons:
            centroid = polygon.centroid
            area = polygon.area
            wkt_geom = polygon.wkt

            row_data = {
                "date": date,
                "centroid_lat": centroid.y,
                "centroid_lon": centroid.x,
                "area": area,
                "geometry_wkt": wkt_geom
            }

            if "name" in gdf.columns:
                row_data["name"] = row.get("name", None)

            processed_rows.append(row_data)

    return pd.DataFrame(processed_rows)


def update_aggregated_csv():
    """Scan all geojson files in OUTPUT_DIR, process them, and update the aggregated CSV."""
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        existing_dates = set(existing_df["date"].dt.date)
    else:
        existing_df = pd.DataFrame()
        existing_dates = set()

    new_dataframes = []

    for file in sorted(os.listdir(OUTPUT_DIR)):
        if file.endswith(".geojson"):
            full_path = os.path.join(OUTPUT_DIR, file)
            file_date = extract_date_from_filename(file)

            if file_date in existing_dates:
                logger.info(f"Skipping {file} (already processed)")
                continue

            logger.info(f"Processing {file} for CSV aggregation...")
            df = process_geojson(full_path)
            if not df.empty:
                new_dataframes.append(df)

    if new_dataframes:
        new_df = pd.concat(new_dataframes, ignore_index=True)
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        combined_df.to_csv(CSV_PATH, index=False)
        logger.info(f"✅ Updated CSV saved to {CSV_PATH}")
    else:
        if not os.path.exists(CSV_PATH):
            logger.warning("⚠️ No data found to create initial CSV.")
        else:
            logger.info("No new data to add.")


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Make API request
    logger.info("Making API request...")
    raw_data = make_api_request()

    # Process data
    logger.info("Processing data...")
    processed_data = process_data(raw_data)

    # Create GeoDataFrame
    logger.info("Creating GeoDataFrame...")
    gdf = create_geodataframe(processed_data)

    # Export as GeoJSON
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    logger.info(f"Exporting data to {output_path}...")
    gdf.to_file(output_path, driver="GeoJSON")

    # Update aggregated CSV
    logger.info("Running CSV aggregation...")
    update_aggregated_csv()

    logger.info("Data update completed successfully.")


if __name__ == "__main__":
    main()
