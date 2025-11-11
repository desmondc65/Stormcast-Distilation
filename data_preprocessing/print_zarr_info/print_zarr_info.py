#!/usr/binbin/env python

"""
A script to inspect a Zarr store and print all metadata for all
groups and arrays, including shape, chunks, dtype, compressor,
and user attributes.
"""

import sys
import zarr
import json

def print_attributes(attrs, indent):
    """Helper function to pretty-print an attributes dictionary."""
    if not attrs:
        print(f"{indent}  (No attributes)")
        return
    for key, value in attrs.items():
        # Pretty print complex values (like dicts/lists)
        if isinstance(value, (dict, list, tuple)):
            try:
                # Use JSON to format it nicely
                value_str = json.dumps(value, indent=2, default=str)
                # Indent the multi-line JSON
                value_str = '\n'.join([f"{indent}    {line}" for line in value_str.splitlines()])
                print(f"{indent}  - {key}:\n{value_str}")
            except Exception:
                # Fallback for un-serializable objects
                print(f"{indent}  - {key}: {value} (un-serializable)")
        else:
            print(f"{indent}  - {key}: {value}")

def print_zarr_info(name, obj, indent_level=0):
    """
    Recursively prints information for a Zarr Group or Array.
    """
    indent = '  ' * indent_level

    if isinstance(obj, zarr.Group):
        print(f"\n{indent}--- Group: {name} ---")
        print(f"{indent}Path: {obj.path}")
        print(f"{indent}Members: {list(obj.keys())}")
        
        print(f"{indent}Attributes:")
        print_attributes(obj.attrs.asdict(), indent + '  ')

        # Recurse into members
        # Changed from obj.items() for compatibility with older zarr versions
        for child_name in obj.keys():
            child_obj = obj[child_name]
            print_zarr_info(child_name, child_obj, indent_level + 1)

    elif isinstance(obj, zarr.Array):
        print(f"\n{indent}--- Array: {name} ---")
        print(f"{indent}Path: {obj.path}")
        print(f"{indent}Shape: {obj.shape}")
        print(f"{indent}Chunks: {obj.chunks}")
        print(f"{indent}DType: {obj.dtype}")
        
        # Handle different zarr versions
        try:
            if hasattr(obj, 'compressors'):
                print(f"{indent}Compressor(s): {obj.compressors}")
            else:
                print(f"{indent}Compressor: {obj.compressor}")
        except Exception as e:
            print(f"{indent}Compressor: (unable to retrieve - {e})")
            
        print(f"{indent}Fill Value: {obj.fill_value}")
        print(f"{indent}Order: {obj.order}")
        print(f"{indent}Filters: {obj.filters}")
        print(f"{indent}Num Bytes: {obj.nbytes:,} (uncompressed)")

        # Handle formatting error for nbytes_stored (which can be None)
        nbytes_stored_val = obj.nbytes_stored
        nbytes_stored_str = "N/A"
        if nbytes_stored_val is not None:
            try:
                nbytes_stored_str = f"{nbytes_stored_val:,} (compressed)"
            except (TypeError, ValueError):
                # Fallback if formatting fails for some other reason
                nbytes_stored_str = f"{nbytes_stored_val} (compressed, unformattable)"
        print(f"{indent}Stored Bytes: {nbytes_stored_str}")
        
        print(f"{indent}Read-only: {obj.read_only}")
        
        print(f"{indent}Attributes:")
        print_attributes(obj.attrs.asdict(), indent + '  ')
            
    else:
        print(f"{indent}Unknown object: {name} (Type: {type(obj)})")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_zarr.py <path_to_zarr_store>")
        print("Example: python inspect_zarr.py my_data.zarr")
        sys.exit(1)

    store_path = sys.argv[1]
    
    try:
        # Open the store in read-only mode
        store = zarr.open(store_path, mode='r')
        
        print(f"Inspecting Zarr store at: {store_path}")
        print("=" * (20 + len(store_path)))
        
        # Start the recursive print from the root group
        print_zarr_info('/', store, indent_level=0)
        
    except (FileNotFoundError, IOError, KeyError) as e:
        print(f"Error: Path not found or invalid. '{store_path}' does not exist or is not a Zarr store.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening or inspecting Zarr store '{store_path}':")
        print(f"{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

