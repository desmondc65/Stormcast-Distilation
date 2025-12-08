#!/bin/bash

python3 data_preprocessing/inspect_tools/inspect_rwrf.py /home/master/13/dczy/code/stormcast-ncdr/data/NC/RWRF/RWRF_2019-08/2019-08-01_00/wrfout_d01_2019-08-01_00_interp > /home/master/13/dczy/code/stormcast-ncdr/data_preprocessing/inspect_tools/rwrf_old.log
python3 data_preprocessing/inspect_tools/inspect_rwrf.py /home/master/13/dczy/code/stormcast-ncdr/data/stormcast_nano5_inference/data_ncdr/RWRF/2025120300/wrfout_d02_2025-12-03_00:20:00 > /home/master/13/dczy/code/stormcast-ncdr/data_preprocessing/inspect_tools/rwrf_new.log