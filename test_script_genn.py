import cv2
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from tonic.slicers import slice_events_by_time

from event_downsampling_genn import EventDownsampleGeNN

"""
Dummy event data
"""
names = ["x", "y", "p", "t"]
formats = ['i4', 'i4', 'i4', 'i4']

# Dummy events (time is in microseconds)
dummy_events = {"events": np.array([(240, 300, 1, 600), (500, 248, 0, 1200), (450, 212, 1, 1800)], 
                                     dtype=list(zip(names, formats))),
                "time_res": 'us'}

# DVSGesture data
dvs_gesture_events_path = 'C:/Users/anind/csdp/data/DVSGesture/ibmGestureTrain/user01_lab/0.npy'
dvs_gesture_events = np.load(dvs_gesture_events_path)

dvs_gesture_events = {"events": unstructured_to_structured(dvs_gesture_events, 
                                                           dtype=np.dtype({'names': names, 
                                                                           'formats': formats})),
                      "time_res": 'ms'}

events = dvs_gesture_events["events"].copy()
time_res = dvs_gesture_events["time_res"]

original_resolution = (128, 128) # h, w
ds_size = (128, 128) # h, w

DT = 1 # milliseconds

"""
Setup GeNN Model
"""
time_window_size = DT

# Initialise GeNN model for event downsampling
use_gpu = True # Set to False to use CPU

genn_downsampler = EventDownsampleGeNN(
    sensor_size=(*original_resolution, 2),
    target_size=ds_size,
    dt=DT,
    tau_theta=420, # milliseconds
    tau_accumulator=16, # milliseconds
    beta=1.2,
    use_gpu=use_gpu,
)

"""
Main Processing Loop
"""

def standardise_time(time_resolution):
    # If events are in microseconds, divide timestamps by 1000 
    # to convert them to milliseconds
    if time_resolution != 'ms':
        return 1000
    else:
        return 1

time_factor = standardise_time(time_resolution=time_res)

events["t"] = (events["t"] / time_factor).astype(int)

num_frames = int((events['t'][-1] / time_factor) + 1)

for t in range(num_frames):
    
    # Temporally sliced events
    sliced_events = slice_events_by_time(events,
                                         time_window=time_window_size,
                                         start_time=t * time_window_size,
                                         end_time=time_window_size * (t + 1))[0]
    
    # Process events through GeNN model
    downsampled_events = genn_downsampler.process_events(sliced_events, iteration=t)
    
    print(t, downsampled_events)
    
    """
    Visualisation with separate polarities
    """
    if downsampled_events.size != 0:
        
        height, width = ds_size
        frame = np.zeros((height, width, 3))
        
        coordinates = np.vstack((downsampled_events['y'], 
                                 downsampled_events['x']))
        
        for p in range(2):
            polarity_indices = np.where(downsampled_events['p'] == p)[0]
            y, x = np.take(coordinates, polarity_indices, axis=1)
            
            # Frames are in BGR format
            frame[...,2*p] = 255 * np.histogram2d(y, x, [np.arange(height+1), np.arange(width+1)])[0]
        
    else:
        frame = np.zeros(ds_size)
    
    resized_frame = cv2.resize(frame, original_resolution)
    cv2.imshow('downsampled_events', resized_frame.astype(np.uint8))
    cv2.waitKey(1)

cv2.destroyAllWindows()