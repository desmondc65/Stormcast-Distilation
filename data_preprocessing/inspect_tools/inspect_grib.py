import pygrib
import sys

def inspect_grib_file(filepath):
    print(f"--- Opening: {filepath} ---")
    
    try:
        grbs = pygrib.open(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # 0. Overall file summary
    messages = list(grbs)
    total_messages = len(messages)
    unique_names = set(m.shortName for m in messages)
    unique_levels = set((m.level, m.typeOfLevel) for m in messages)
    unique_dates = set(m.dataDate for m in messages)
    unique_times = set(m.dataTime for m in messages)
    
    print(f"\n{'='*60}")
    print(f"OVERALL FILE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Messages: {total_messages}")
    print(f"  Unique Variables: {len(unique_names)} -> {sorted(unique_names)}")
    print(f"  Unique Levels: {len(unique_levels)} -> {sorted(unique_levels)}")
    print(f"  Unique Dates: {sorted(unique_dates)}")
    print(f"  Unique Times: {sorted(unique_times)}")
    
    # Get data shape from first message
    if messages:
        sample_data = messages[0].values
        print(f"  Grid Shape (per message): {sample_data.shape}")
        print(f"  Overall Data Shape: ({total_messages}, {sample_data.shape[0]}, {sample_data.shape[1]})")
    print(f"{'='*60}\n")

    # 1. Loop through messages to get a summary
    # A GRIB file is a stream of binary messages. We iterate through them.
    for i, grb in enumerate(messages):
        print(f"\nMessage {i+1}:")
        print(f"  Name: {grb.name}")
        print(f"  ShortName: {grb.shortName}")
        print(f"  Level: {grb.level} ({grb.typeOfLevel})")
        print(f"  Date/Time: {grb.dataDate} {grb.dataTime}")
        print(f"  Forecast Time: {grb.stepRange}")
        # print(f"  Units: {grb.units}")
        # print(f"  Grid Type: {grb.gridType}")
        # print(f"  Ni x Nj: {grb.Ni} x {grb.Nj}")
        # print(f"  Lat Range: {grb.latitudeOfFirstGridPointInDegrees} to {grb.latitudeOfLastGridPointInDegrees}")
        # print(f"  Lon Range: {grb.longitudeOfFirstGridPointInDegrees} to {grb.longitudeOfLastGridPointInDegrees}")
        # print(f"  Parameter ID: {grb.paramId}")
        # print(f"  Centre: {grb.centre}")
        # print(f"  Analysis/Forecast: {grb.dataType}")
        # print(f"  Missing Value: {grb.missingValue}")
        
        # 2. Extracting Data Values (The Grid)
        data = grb.values  # This returns a numpy array
        lats, lons = grb.latlons() # Returns lat/lon grids
        
        # print lat and lon range
        print(f"  Latitude Range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"  Longitude Range: {lons.min():.2f} to {lons.max():.2f}")  
        
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