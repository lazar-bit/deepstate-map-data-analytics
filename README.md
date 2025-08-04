[![Python - 3.12.4](https://img.shields.io/badge/Python-3.12.4-f4d159)](https://www.python.org/downloads/release/python-3124/)
[![Update Data](https://github.com/cyterat/deepstate-map-data/actions/workflows/update.yml/badge.svg)](https://github.com/cyterat/deepstate-map-data/actions/workflows/update.yml)

# 🛰️ DeepState Map Data — Enhanced Fork

> **Forked from [deepstate-map-data](https://github.com/cyterat/deepstate-map-data)** by [cyterat](https://github.com/cyterat/)  
> Modified and extended by [Zsolt Lazar](https://github.com/lazar-bit)

This repository collects daily GeoJSON data from DeepStateMap.Live, representing current Russian-occupied areas in Ukraine.  
This fork introduces automation enhancements and adds a cumulative CSV aggregation feature, making the data more accessible for OSINT workflows and time-series analysis.

---

## 📁 Data Structure

The `data/` folder contains up-to-date Multipolygons in GeoJSON format.

- **Filename format:** `deepstatemap_data_<update_date>.geojson`
- **Update frequency:** Daily, at 03:00 UTC via GitHub Actions

> **Note:** The Kaggle dataset upload process creates a temporary `kaggle_dataset` folder inside the GitHub Actions runner environment to prepare files for upload. This folder **is not created or committed** inside GitHub repository and does not affect repo structure.

---

## 🛠️ Enhancements by Zsolt Lazar

- Added automatic **daily CSV aggregation** across all GeoJSON files
- Ensures a continuously growing CSV file for easier analysis
- **Automatic upload of the updated dataset to [Kaggle](https://www.kaggle.com/datasets/zsoltlazar/automated-deepstatemap-occupied-areas/data)** for easy access and sharing
- Structured for **OSINT dashboards** and **geospatial workflows**
- Improved code readability and modularity

---

## 📜 License and Attribution

This project is based on the work by [Original Author](https://github.com/ORIGINAL_AUTHOR).  
All original licensing terms apply and are retained.  
This fork is maintained for educational, analytical, and academic purposes.
