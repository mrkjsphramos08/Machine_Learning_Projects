# Dataset Provenance: California Housing

- **Source**: `sklearn.datasets.fetch_california_housing` (derived from the 1990 U.S. Census data)
- **Retrieved**: 2026-09-23
- **Licence**: Public Domain / BSD-3-Clause (Scikit-Learn distribution)
- **Rows / Columns**: 20,640 rows / 9 columns (8 features + 1 target)
- **SHA256**: `7192AF0BF33E94141194CFD91958C62D66262DC40C07CC2E6FBE0B2D02B0E1B3`
- **Target Column**: `MedHouseVal` (Median house value for California districts, in units of 100,000 USD)
- **Features**:
  - `MedInc`: Median income in block group
  - `HouseAge`: Median house age in block group
  - `AveRooms`: Average number of rooms per household
  - `AveBedrms`: Average number of bedrooms per household
  - `Population`: Block group population
  - `AveOccup`: Average number of household members
  - `Latitude`: Block group latitude
  - `Longitude`: Block group longitude
- **Notes**: Zero missing values in raw form. Target values are capped at 5.0 ($500,000) in the original census collection.
