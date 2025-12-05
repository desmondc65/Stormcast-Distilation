import pygrib
import sys

def inspect_grib_file(filepath):
    print(f"--- Opening: {filepath} ---")
    
    try:
        grbs = pygrib.open(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # 1. Loop through messages to get a summary
    # A GRIB file is a stream of binary messages. We iterate through them.
    for i, grb in enumerate(grbs):
        print(f"\nMessage {i+1}:")
        print(f"  Name: {grb.name}")
        print(f"  ShortName: {grb.shortName}")
        print(f"  Level: {grb.level} ({grb.typeOfLevel})")
        print(f"  Date/Time: {grb.dataDate} {grb.dataTime}")
        print(f"  Forecast Time: {grb.stepRange}")
        
        # 2. Extracting Data Values (The Grid)
        data = grb.values  # This returns a numpy array
        lats, lons = grb.latlons() # Returns lat/lon grids
        
        print(f"  Data Shape: {data.shape}")
        print(f"  Min/Max Value: {data.min():.4f} / {data.max():.4f}")
        
        # 3. Accessing ALL Metadata Keys
        # GRIB files have hundreds of keys. Uncomment below to see everything.
        # keys = grb.keys()
        # for key in keys:
        #     try:
        #         print(f"    {key}: {grb[key]}")
        #     except:
        #         pass

    grbs.close()

if __name__ == "__main__":
    # Replace with your .grib or .grib2 file path
    file_path = "/home/master/13/dczy/code/stormcast-ncdr/data/stormcast_nano5_inference/data_ncdr/Global/2025120300/EC-pangu_2025120300-0.grb" 
    inspect_grib_file(file_path)