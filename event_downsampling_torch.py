import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from tonic.functional.to_frame import to_frame_numpy
import torch

def event_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, 
                          tau_theta: float, tau_accumulator: float, beta: float, iteration: int,
                          iter_args: tuple, gpu):
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
    
    if iteration == 0:
        assert {"x", "y"} <= set(events.dtype.names)

        # Normalise event key names for compatibility
        event_key_map = {"timestamp": "t", "polarity": "p"}
        event_key_names = list(events.dtype.names)

        for old_key, new_key in event_key_map.items():
            if old_key in event_key_names:
                event_key_names[event_key_names.index(old_key)] = new_key

        events.dtype.names = tuple(event_key_names)
        assert dt is not None

    # Convert integer timestamps (assumed to be in microseconds) to milliseconds
    if np.issubdtype(events["t"].dtype, np.integer):
        dt, tau_theta, tau_accumulator, beta = [x * 1000 for x in (dt, tau_theta, tau_accumulator, beta)]

    accumulator, theta = iter_args
    
    events = events.copy()
    
    # Spatial downsampling
    spatial_factor = np.asarray(target_size) / sensor_size[:-1] # h,w

    events["x"] = (events["x"] * spatial_factor[0] - 1).clip(min=0)
    events["y"] = (events["y"] * spatial_factor[1] - 1).clip(min=0)
    
    # Compute all histograms at once
    all_frame_histograms = to_frame_numpy(events, sensor_size=(*target_size[::-1], 2), time_window=dt, 
                                          start_time=iteration*dt, end_time=(iteration+1)*dt) # w,h
    
    # Subtract the channels for ON/OFF differencing
    frame_histogram = (all_frame_histograms[:, 1] - all_frame_histograms[:, 0])[0]
    
    exp_decay_acc = torch.tensor(np.exp(-dt / tau_accumulator), device=gpu)
    exp_decay_theta = torch.tensor(np.exp(-dt / tau_theta), device=gpu)
    
    events_new = [[] for i in range(2)]
    
    frame_histogram = torch.tensor(frame_histogram.flatten(), device=gpu) # Move histogram to GPU
    
    # Update accumulator
    accumulator.mul_(1 - exp_decay_acc).add_(frame_histogram.mul(exp_decay_acc))
    
    target_size_unravelled = np.prod(target_size)
    
    coordinates_pos = torch.nonzero(torch.maximum(accumulator > theta[...,0], torch.zeros(target_size_unravelled).to(gpu)))
    coordinates_neg = torch.nonzero(torch.maximum(-accumulator > theta[...,1], torch.zeros(target_size_unravelled).to(gpu)))
            
    theta *= np.exp(-dt / tau_theta)
    
    if np.logical_or(coordinates_pos.numel(), coordinates_neg.numel()):
        for c, coordinate in enumerate([coordinates_neg, coordinates_pos]):
            
            # rows, cols = (coordinate[:, 0], coordinate[:,1]) # For readability
            # coords = rows * target_size[1] + cols
            
            # Update threshold
            theta[coordinate,c] += beta * torch.abs(accumulator[coordinate]) * (1 - np.exp(-dt / tau_theta))
            
            # Reset spiking neurons to zero
            accumulator[coordinate] = 0
            
            # Restructure events
            num_events = coordinate.size(0)
            coord_copy = coordinate.clone().detach().cpu().numpy()
            
            coord_2D = np.unravel_index(coord_copy, target_size)
            
            events_new[c].append((coord_2D[0], coord_2D[1], 
                              c * np.ones((num_events, 1)).astype(dtype=bool), 
                              (iteration * dt) * np.ones((num_events, 1))))
            
            events_new[c] = np.concatenate(events_new[c])
            
        events_new = np.concatenate(events_new, axis=1)[...,0]
        
        names = ["x", "y", "p", "t"]
        formats = ['i4', 'i4', 'i4', 'i4']
        
        dtype = np.dtype({'names': names, 'formats': formats})
        
        events_new = unstructured_to_structured(events_new.T, dtype=dtype)
    else:
        events_new = np.asarray(events_new)
    
    return (accumulator, theta), events_new