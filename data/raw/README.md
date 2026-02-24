# Raw Data

Source dataset: `Agri_yield_prediction.csv` (36,000 rows, 46 columns).

The CSV is not committed to the repository (gitignored to keep history small).
Place the file at `data/raw/Agri_yield_prediction.csv` before running the
notebooks.

## Columns

- Climate: Temperature, Humidity, Rainfall, Solar_Radiation, Wind_Speed
- Soil: pH, EC, OC, N, P, K, Ca, Mg, S, micronutrients (Cu, Zn, Fe, Mn, B, Mo), CEC
- Texture: Sand, Silt, Clay, Bulk_Density, Water_Holding_Capacity
- Terrain: Slope, Aspect, Elevation
- Vegetation indices: NDVI, EVI, LAI, Chlorophyll, GDD
- Categorical: Crop_Type, Soil_Type, Fertilizer_Type, Pesticide_Usage,
  Growth_Stage, Region, Season
- Time: Planting_Date, Harvest_Date, Year, Irrigation_Frequency
- Target: Yield (tons/ha)

No missing values, no duplicates.
