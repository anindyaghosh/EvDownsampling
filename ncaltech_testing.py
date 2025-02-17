import cv2
import numpy as np
import os

import tonic
from tonic.datasets import NCALTECH101
from event_based_downsampling import event_downsample

ncaltech = NCALTECH101("data")
events, label = ncaltech[2000]
print(label)

height, width = (events['y'].max()+1, events['x'].max()+1)
spatial_downsampling_factor = 2
downsampled_height, downsampled_width = (np.asarray([height, width]) / 
                                         spatial_downsampling_factor).astype(int)

def visualisation(events, time_window, downsampled_resolution, resolution):
    events['t'] = (events['t'] / time_window).astype(int)
    
    num_frames = int(events['t'][-1]) + 1
    
    height, width = resolution
    
    if events['x'].max()+1 == width:
        downsampled_height, downsampled_width = resolution
    else:
        downsampled_height, downsampled_width = downsampled_resolution
    
    for t in range(num_frames):
        frame = np.zeros((downsampled_height, downsampled_width, 3))
        
        indices = np.where(events['t'] == t)[0]
        sliced_events = np.take(events, indices)
        
        coordinates, polarity = np.vstack((sliced_events['x'], sliced_events['y'])), sliced_events['p']
        
        for p in range(2):
            polarity_indices = np.where(polarity == p)[0]
            x, y = np.take(coordinates, polarity_indices, axis=1)
            
            # Frames are in BGR format
            frame[...,p] = 255 * np.histogram2d(y, x, [np.arange(downsampled_height+1), 
                                                       np.arange(downsampled_width+1)])[0]
        
        frame[...,[1, 2]] = frame[...,[2, 1]]
        frame = cv2.resize(frame, np.array([width, height])*4)
            
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()

downsampled_events = event_downsample(events, (width, height, 2), 
                                      (downsampled_width, downsampled_height), 
                                      dt=1,
                                      tau_theta=16,
                                      tau_accumulator=420,
                                      beta=1.2)

# Change first parameter to 'events' to visualise original event stream
visualisation(downsampled_events, 
              time_window=1000, 
              downsampled_resolution=(downsampled_height, downsampled_width), 
              resolution=(height, width))

print(f'{len(downsampled_events)/len(events) * 100:.1f}%')