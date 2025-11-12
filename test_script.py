import cv2
import numpy as np
from tonic.slicers import slice_events_by_time
import torch

from event_downsampling_torch import event_downsample

original_resolution = (480, 640) # h, w
ds_size = (48, 64) # h, w

DT = 1 # milliseconds

"""
Dummy event data
"""
names = ["x", "y", "p", "t"]
formats = ['i4', 'i4', 'i4', 'i4']

# Dummy events (time is in microseconds)
dummy_events = np.array([(240, 300, 1, 600), (500, 248, 0, 1200), (450, 212, 1, 1800)], 
                        dtype=list(zip(names, formats)))

"""
Setup
"""
time_window_size = 1000 * DT

# Replace 'cuda' with 'cpu' to run in cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

iter_accumulator = torch.zeros(np.prod(ds_size), dtype=torch.float16).to(device)
iter_theta = torch.zeros((np.prod(ds_size), 2), dtype=torch.float16).to(device)

iter_args = (iter_accumulator, iter_theta)

"""
Main
"""

num_frames = int(dummy_events['t'][-1]/ 1000)

for t in range(num_frames):
    
    # Temporally sliced events
    sliced_events = slice_events_by_time(dummy_events, 
                                         time_window=time_window_size, 
                                         start_time=t*time_window_size, 
                                         end_time=time_window_size*(t+1))[0]
    
    with torch.no_grad():
        # Call EvDownsampling
        iter_args, downsampled_events = event_downsample(sliced_events, 
                                                         (*original_resolution, 2), 
                                                         ds_size, 
                                                         dt=DT, 
                                                         tau_theta=420, 
                                                         tau_accumulator=16, 
                                                         beta=1.2, 
                                                         iteration=t, 
                                                         iter_args=iter_args,
                                                         gpu=device)
    
    """
    Visualisation with merged polarities
    """
    if downsampled_events.size != 0:
        frame = np.histogram2d(downsampled_events['y'], 
                               downsampled_events['x'], 
                               [np.arange(ds_size[0]), np.arange(ds_size[1])])[0]
    else:
        frame = np.zeros(ds_size)
    
    cv2.imshow('downsampled_events', (frame*255).astype(np.uint8))
    cv2.waitKey(1)
    
cv2.destroyAllWindows()