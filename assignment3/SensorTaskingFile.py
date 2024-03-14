import numpy as np
import json
from datetime import datetime

###############################################################################
# This script will generate the sensor tasking file in correct format.
#
# Edit the fields in this section based on your desired sensor tasks.
#
###############################################################################

# File name should include group number
json_file = 'group7_sensor_tasking_file.json'

# Sensor type is a string, either 'radar' or 'optical'
sensor_type = 'radar'
#sensor_type = 'optical'

# Sensor location
latitude_radians = np.deg2rad(10.)
longitude_radians = np.deg2rad(20.)
height_meters = 30.

# Indicate start times and object IDs for each task.
# You must ensure that sensor constraints are met.
# If an object is not visible or violates, no measurements will be generated.

# All sensor tasks will track the object for 5 minutes, producing measurements
# at an interval of 10 seconds. If the visible pass is shorter than 5 minutes
# from the start time, measurements will be computed until the object is out
# of view.

# Start time should be given as a datetime object in UTC
# Round times to nearest second

# Task 1
task1_start_UTC = datetime(2000, 1, 1, 12, 0, 0)
task1_object_id = 10001

# Task 2
task2_start_UTC = datetime(2000, 1, 2, 12, 0, 0)
task2_object_id = 10002

# Task 3
task3_start_UTC = datetime(2000, 1, 3, 12, 0, 0)
task3_object_id = 10003

# Task 4
task4_start_UTC = datetime(2000, 1, 4, 12, 0, 0)
task4_object_id = 10004

# Task 5
task5_start_UTC = datetime(2000, 1, 5, 12, 0, 0)
task5_object_id = 10005

# Task 6
task6_start_UTC = datetime(2000, 1, 6, 12, 0, 0)
task6_object_id = 10006

# Task 7
task7_start_UTC = datetime(2000, 1, 7, 12, 0, 0)
task7_object_id = 10007

# Task 8
task8_start_UTC = datetime(2000, 1, 8, 12, 0, 0)
task8_object_id = 10008

# Task 9
task9_start_UTC = datetime(2000, 1, 9, 12, 0, 0)
task9_object_id = 10009

# Task 10
task10_start_UTC = datetime(2000, 1, 10, 12, 0, 0)
task10_object_id = 10010




###############################################################################
# This code will generate the tasking file in correct JSON format
#
# DO NOT EDIT
#
###############################################################################




tasking_dict = {}
tasking_dict['sensor_type'] = sensor_type
tasking_dict['latitude_radians'] = latitude_radians
tasking_dict['longitude_radians'] = longitude_radians
tasking_dict['height_meters'] = height_meters

tasking_dict['1'] = {}
tasking_dict['1']['start'] = task1_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['1']['obj_id'] = task1_object_id

tasking_dict['2'] = {}
tasking_dict['2']['start'] = task2_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['2']['obj_id'] = task2_object_id

tasking_dict['3'] = {}
tasking_dict['3']['start'] = task3_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['3']['obj_id'] = task3_object_id

tasking_dict['4'] = {}
tasking_dict['4']['start'] = task4_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['4']['obj_id'] = task4_object_id

tasking_dict['5'] = {}
tasking_dict['5']['start'] = task5_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['5']['obj_id'] = task5_object_id

tasking_dict['6'] = {}
tasking_dict['6']['start'] = task6_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['6']['obj_id'] = task6_object_id

tasking_dict['7'] = {}
tasking_dict['7']['start'] = task7_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['7']['obj_id'] = task7_object_id

tasking_dict['8'] = {}
tasking_dict['8']['start'] = task8_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['8']['obj_id'] = task8_object_id

tasking_dict['9'] = {}
tasking_dict['9']['start'] = task9_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['9']['obj_id'] = task9_object_id

tasking_dict['10'] = {}
tasking_dict['10']['start'] = task10_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['10']['obj_id'] = task10_object_id


with open(json_file, 'w') as f:
    json.dump(tasking_dict, f)





























