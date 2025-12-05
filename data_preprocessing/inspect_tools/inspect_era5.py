import netCDF4
import sys
import numpy as np

def print_structure(nc_object, indent=0):
    """
    Recursively prints the structure of a NetCDF4 object (Dataset or Group).
    """
    prefix = "  " * indent
    
    # 1. Print Global/Group Attributes
    print(f"{prefix}--- Attributes (Global/Group) ---")
    for attr in nc_object.ncattrs():
        val = getattr(nc_object, attr)
        # Truncate really long attributes so terminal doesn't flood
        str_val = str(val)
        if len(str_val) > 80:
            str_val = str_val[:77] + "..."
        print(f"{prefix}  {attr}: {str_val}")

    # 2. Print Dimensions
    if len(nc_object.dimensions) > 0:
        print(f"{prefix}--- Dimensions ---")
        for dim_name, dim_obj in nc_object.dimensions.items():
            print(f"{prefix}  {dim_name}: size={len(dim_obj)} {'(unlimited)' if dim_obj.isunlimited() else ''}")

    # 3. Print Variables
    if len(nc_object.variables) > 0:
        print(f"{prefix}--- Variables ---")
        for var_name, var_obj in nc_object.variables.items():
            print(f"{prefix}  Name: {var_name}")
            print(f"{prefix}    Dimensions: {var_obj.dimensions}")
            print(f"{prefix}    Shape: {var_obj.shape}")
            print(f"{prefix}    Dtype: {var_obj.dtype}")
            
            # Print Variable Attributes
            for attr in var_obj.ncattrs():
                print(f"{prefix}    - {attr}: {getattr(var_obj, attr)}")
            print("")

    # 4. Handle Groups (NetCDF4 supports folder-like hierarchy)
    # Most simple weather files don't use this, but complex ones do.
    if len(nc_object.groups) > 0:
        print(f"{prefix}--- Groups ---")
        for group_name, group_obj in nc_object.groups.items():
            print(f"\n{prefix}Group: {group_name}")
            print_structure(group_obj, indent + 1)

def inspect_nc_file(filepath):
    try:
        # 'r' is for read-only
        with netCDF4.Dataset(filepath, 'r') as ds:
            print(f"=== File: {filepath} ===")
            print(f"Format: {ds.data_model}")
            print_structure(ds)
            
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except OSError as e:
        print(f"Error: Could not open file (is it a valid NetCDF?): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_nc.py <filename.nc>")
        # For testing, you can hardcode a path below if you prefer
        # inspect_nc_file("sample.nc") 
    else:
        inspect_nc_file(sys.argv[1])