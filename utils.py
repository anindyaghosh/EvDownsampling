import numpy as np

def changeEventDtypeNames(events):
    event_key_names = list(events.dtype.names)
    try:
        event_key_names[event_key_names.index("timestamp")] = "t"
        event_key_names[event_key_names.index("polarity")] = "p"
    except:
        pass
    events.dtype.names = tuple(event_key_names)

def build_frames(events, camera_name, time_resolution, output_resolution):
    
    events = events.copy()
    
    # Called to standardise key names
    changeEventDtypeNames(events)
    
    num_frames_integrated = time_resolution # ms
    
    # Temporally downsample to 1 ms resolution using publisher rate
    events['t'] = (events['t'] / (num_frames_integrated * 10**3)).astype(int)
    
    num_frames = events['t'][-1] + 1
    
    width, height = output_resolution
    
    frames = np.zeros((num_frames, height, width, 3))
    
    for t in range(num_frames):
        frame = np.zeros((height, width, 3))
        
        indices = np.where(events['t'] == t)[0]
        sliced_events = np.take(events, indices)
        
        coordinates, polarity = np.vstack((sliced_events['x'], sliced_events['y'])), sliced_events['p']
        
        for p in range(2):
            polarity_indices = np.where(polarity == p)[0]
            x, y = np.take(coordinates, polarity_indices, axis=1)
            
            # Frames are in BGR format
            frame[...,p] = 255 * np.histogram2d(y, x, [np.arange(height+1), np.arange(width+1)])[0]
            
        frames[t] += frame
        
    return frames