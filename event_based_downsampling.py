import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from tonic.functional.to_frame import to_frame_numpy

def event_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, 
                          tau_theta: float, tau_accumulator: float, beta: float):
    """Spatio-temporally downsample events using a LIF neuron with an adaptive threshold.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): temporal resolution of events in milliseconds.
        tau_theta (float): time constant of adaptive threshold (in milliseconds).
        tau_accumulator (float): time constant of accumulator (in milliseconds).
        beta (float): constant to scale input to threshold after each timestep 
                      (function of dt in milliseconds).
        
    Returns:
        the spatio-temporally downsampled input events.
    """
    
    assert "x" and "y" in events.dtype.names
    
    # Some DVS cameras output "t" and "p" instead of "timestamp" and "polarity" respectively.
    try:
        assert "t" in events.dtype.names
    except:
        assert "timestamp" in events.dtype.names
        event_key_names = list(events.dtype.names)
        event_key_names[event_key_names.index("timestamp")] = "t"
        events.dtype.names = tuple(event_key_names)
        
    try:
        assert "p" in events.dtype.names
    except:
        assert "polarity" in events.dtype.names
        event_key_names = list(events.dtype.names)
        event_key_names[event_key_names.index("polarity")] = "p"
        events.dtype.names = tuple(event_key_names)
        
    assert dt is not None
    
    events = events.copy()
    
    if np.issubdtype(events["t"].dtype, np.integer):
        # Assumes raw event times are in microseconds
        dt *= 1000
        tau_theta *= 1000
        tau_accumulator *= 1000
        beta *= 1000
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]
    
    # Get time of first event
    first_event_time = events["t"][0]
    
    # Compute all histograms at once
    all_frame_histograms = to_frame_numpy(events, sensor_size=(*target_size, 2), time_window=dt)
    
    # Subtract the channels for ON/OFF differencing
    frame_histogram_diffs = all_frame_histograms[:, 1] - all_frame_histograms[:, 0]
    
    target_size = np.flip(target_size)
    
    # Initialise accumulator and threshold
    accumulator = np.zeros(target_size, dtype=np.float16)
    theta = np.zeros((np.prod(target_size), 2), dtype=np.float16)
    
    events_new = []
    
    for time, frame_histogram in enumerate(frame_histogram_diffs):
        
        # Update accumulator
        accumulator = (1-np.exp(-dt / tau_accumulator))*accumulator + frame_histogram*np.exp(-dt / tau_accumulator)
        
        theta_unravelled = np.reshape(theta, (*target_size, 2))
            
        coordinates_pos = np.stack(np.nonzero(np.maximum(accumulator > theta_unravelled[...,1], 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-accumulator > theta_unravelled[...,0], 0))).T
                
        theta *= np.exp(-dt / tau_theta)
        
        if np.logical_or(coordinates_pos.size, coordinates_neg.size).sum():
            for c, coordinate in enumerate([coordinates_neg, coordinates_pos]):
                
                coords = np.ravel_multi_index(np.split(coordinate, 2, axis=1), target_size)
                
                # Update threshold
                theta[np.squeeze(coords),c] += beta * np.abs(accumulator[tuple(coordinate.T)]) * (1 - np.exp(-dt / tau_theta))
                
                # Reset spiking neurons to zero
                accumulator[tuple(coordinate.T)] = 0
                
                # Restructure events
                events_new.append(np.column_stack((np.flip(coordinate, axis=1),
                                                   c * np.ones((coordinate.shape[0], 1)).astype(dtype=bool), 
                                                   (time * dt) * np.ones((coordinate.shape[0], 1)))))
        
    events_new = np.concatenate(events_new.copy())
    
    names = ["x", "y", "p", "t"]
    formats = ['i4', 'i4', 'i4', 'i4']
    
    dtype = np.dtype({'names': names, 'formats': formats})
    
    events_new = unstructured_to_structured(events_new.copy(), dtype=dtype)
    # To ensure timestamps are adjusted accordingly
    events_new["t"] += first_event_time
    
    return events_new