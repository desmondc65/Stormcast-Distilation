### High- and Low-Resolution Meteorological Data Configuration

To train the aforementioned model, this project adopts a data configuration strategy of "low-resolution guidance, high-resolution learning," integrating global and regional data:

  * **Low-Resolution Input:**
    Utilizes the **ERA5** reanalysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF) \\citep{hersbach2020era5}.

      * **Resolution:** Spatial resolution is approximately 25 km.
      * **Purpose:** Provides global synoptic-scale meteorological variables (e.g., geopotential height, mean sea level pressure, wind fields, temperature) to serve as large-scale environmental conditioning for the model.

  * **High-Resolution Target:**
    Serves as the learning objective (Ground Truth) for the model, utilizing **RWRF** (Radar-assimilated Weather Research and Forecasting) numerical model outputs with a resolution of 2 to 3 km, along with **QPEPRE** (Quantitative Precipitation Estimation) data.

      * **RWRF:** Provides fine-scale 3D dynamic and thermodynamic structures over the Taiwan region.
      * **QPEPRE:** Provides high-resolution observational ground truth for surface rainfall.
      * **Purpose:** These datasets contain the detailed dynamic and thermodynamic structures shaped by Taiwan's complex terrain, making them crucial for the model to learn localized features.

**Proposed Training Data Configuration**

| Characteristics | Low-Resolution Input | High-Resolution Target |
| :--- | :--- | :--- |
| **Data Source** | **ERA5** (Reanalysis data) | **RWRF** (Numerical model) & **QPEPRE** (Observation) |
| **Spatial Resolution** | \~25 km | 2 - 3 km |
| **Primary Purpose** | Provides large-scale synoptic environmental guidance | Provides localized fine structures and rainfall ground truth |
| **Surface / Single-Level Variables** | **Mean Sea Level Pressure (mslp)**<br>**2-meter Temperature (t2m)**<br>**10-meter U-wind (u10)**<br>**10-meter V-wind (v10)** | **2-meter Temperature (t2m)**<br>**10-meter U-wind (u10)**<br>**10-meter V-wind (v10)**<br>**Radar QPE (qpepre)** |
| **Upper-Air / Multi-Level Variables** | **Specific Humidity (q)**<br>*(1000, 850, 500, 250 hPa)*<br>**Temperature (t)**<br>*(1000, 850, 500, 250 hPa)*<br>**U-wind (u)**<br>*(1000, 850, 500, 250 hPa)*<br>**V-wind (v)**<br>*(1000, 850, 500, 250 hPa)*<br>**Geopotential Height (z)**<br>*(1000, 850, 500, 250 hPa)* | *(No upper-air variables; focuses on surface rainfall and near-surface features)* |

