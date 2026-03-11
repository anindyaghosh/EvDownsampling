import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from tonic.functional.to_frame import to_frame_numpy
from pygenn import (genn_model, create_neuron_model, create_current_source_model)

# Custom LIF neuron model with adaptive threshold and polarity tracking for GeNN
lif_adaptive_model = create_neuron_model(
    "alif_pop", 
    params=["tau_acc", "tau_theta", "beta", "dt"],
    vars=[
        ("V", "scalar"),        # Accumulator (membrane potential)
        ("Theta0", "scalar"),   # Threshold for negative polarity
        ("Theta1", "scalar"),   # Threshold for positive polarity
        ("LastSpikePol", "uint8_t"),  # Track polarity of last spike
    ],
    sim_code= 
    """
        // Decay accumulator (applied at each timestep)
        scalar decay_acc = exp(-dt / tau_acc);
        V *= (1.0 - decay_acc);
        V += Isyn * (1.0 - decay_acc);
        
        // Decay thresholds
        scalar decay_theta = exp(-dt / tau_theta);
        Theta0 *= decay_theta;
        Theta1 *= decay_theta;
    """,
    threshold_condition_code= 
    """
        (V > Theta1) || (-V > Theta0)
    """,
    reset_code=
    """
        // Determine which threshold was crossed and update it
        if (V > Theta1) {
            Theta1 += beta * abs(V) * (1.0 - decay_theta);
            LastSpikePol = 1;  // Positive polarity
        } else if (-V > Theta0) {
            Theta0 += beta * abs(V) * (1.0 - decay_theta);
            LastSpikePol = 0;  // Negative polarity
        }
        // Reset accumulator
        V = 0.0;
    """
)

# Custom current source to inject event histogram values
histogram_input_model = create_current_source_model(
    "histogram_input",
    vars=[("magnitude", "scalar")],
    injection_code= 
    """
        scalar decay_acc = exp(-dt / tau_acc);
        injectCurrent(magnitude * decay_acc);
        magnitude = 0.0;  // Reset after injection
    """,
    params=["tau_acc", "dt"]
    )

class EventDownsampleGeNN:
    """Spatio-temporally downsample events using GeNN-based LIF neurons with adaptive thresholds."""
    
    def __init__(self, sensor_size: tuple, target_size: tuple, dt: float,
                 tau_theta: float, tau_accumulator: float, beta: float,
                 use_gpu: bool = True):
        """
        Initialise the GeNN model for event downsampling.
        
        Parameters:
            sensor_size (tuple): a 3-tuple of x,y,p for original sensor size.
            target_size (tuple): a 2-tuple of x,y denoting downsampled size (width, height).
            dt (float): temporal resolution in milliseconds.
            tau_theta (float): time constant of adaptive threshold (milliseconds).
            tau_accumulator (float): time constant of accumulator (milliseconds).
            beta (float): constant to scale threshold adaptation.
            use_gpu (bool): whether to use GPU acceleration.
        """
        
        self.sensor_size = sensor_size
        self.target_size = target_size
        self.dt = dt
        self.tau_theta = tau_theta
        self.tau_accumulator = tau_accumulator
        self.beta = beta
        self.use_gpu = use_gpu
        
        # Calculate number of neurons (one per downsampled pixel)
        self.num_neurons = int(np.prod(target_size))
        
        # Create GeNN model
        self.model = genn_model.GeNNModel("float", "event_downsample")
        self.model.dt = dt
        
        # Set backend preferences
        if use_gpu:
            try:
                self.model.backend = "cuda"
            except:
                print("CUDA not available, falling back to CPU")
                self.model.backend = "SingleThreadedCPU"
        else:
            self.model.backend = "SingleThreadedCPU"
        
        # Create neuron population with adaptive LIF model
        alif_params = {
            "tau_acc": tau_accumulator,
            "tau_theta": tau_theta,
            "beta": beta,
            "dt": dt
        }
        
        alif_init = {
            "V": 0.0,
            "Theta0": 0.0,
            "Theta1": 0.0,
            "LastSpikePol": 0
        }
        
        self.neurons = self.model.add_neuron_population(
            "neurons", self.num_neurons, lif_adaptive_model,
            alif_params, alif_init
        )
        
        # Enable spike recording
        self.neurons.spike_recording_enabled = True
        
        # Create current source for histogram injection
        cs_params = {
            "tau_acc": tau_accumulator,
            "dt": dt
        }
        
        cs_init = {"magnitude": 0.0}
        
        self.current_source = self.model.add_current_source(
            "histogram_input", histogram_input_model,
            self.neurons, cs_params, cs_init
        )
        
        # Build and load model
        print("Building GeNN model...")
        self.model.build()
        print("Loading GeNN model...")
        self.model.load(num_recording_timesteps=1)
        print("GeNN model ready!")
        
        # Get views for variables
        self.magnitude_view = self.current_source.vars["magnitude"].view
        self.V_view = self.neurons.vars["V"].view
        self.Theta0_view = self.neurons.vars["Theta0"].view
        self.Theta1_view = self.neurons.vars["Theta1"].view
        self.polarity_view = self.neurons.vars["LastSpikePol"].view
        
    def process_events(self, events: np.ndarray, iteration: int):
        """
        Process events for one time step.
        
        Parameters:
            events (ndarray): structured array with fields 'x', 'y', 'p', 't'.
            iteration (int): current iteration/frame number.
            
        Returns:
            downsampled_events (ndarray): spatially and temporally downsampled events.
        """
        # Spatial downsampling
        events = events.copy()
        spatial_factor = np.asarray(self.target_size) / self.sensor_size[:-1] # h, w
        
        events["x"] = (events["x"] * spatial_factor[1]).astype(int).clip(min=0, max=self.target_size[1]-1)
        events["y"] = (events["y"] * spatial_factor[0]).astype(int).clip(min=0, max=self.target_size[0]-1)
        
        # Compute histogram for this time window
        all_frame_histograms = to_frame_numpy(
            events, 
            sensor_size=(*self.target_size[::-1], 2),
            time_window=self.dt,
            start_time=iteration * self.dt,
            end_time=(iteration + 1) * self.dt
        )
        
        # ON - OFF differencing
        frame_histogram = (all_frame_histograms[:, 1] - all_frame_histograms[:, 0])[0]
        frame_histogram_flat = frame_histogram.flatten().astype(np.float32)
        
        # Inject histogram values into neurons
        self.magnitude_view[:] = frame_histogram_flat
        self.current_source.vars["magnitude"].push_to_device()
        
        # Step simulation
        self.model.step_time()
        
        # Pull spike data and neuron states
        self.model.pull_recording_buffers_from_device()
        self.neurons.vars["LastSpikePol"].pull_from_device()
        
        # Get spikes from this timestep
        spike_times, spike_ids = self.neurons.spike_recording_data[0]
        
        # Filter spikes for current timestep
        current_time = iteration * self.dt
        mask = (spike_times >= current_time) & (spike_times < current_time + self.dt)
        spike_ids_current = spike_ids[mask]
        
        if len(spike_ids_current) > 0:
            # Convert flat indices to 2D coordinates
            coords_2d = np.unravel_index(spike_ids_current, self.target_size)
            
            # Get polarities from neuron state
            polarities = self.polarity_view[spike_ids_current].astype(np.int32)
            
            # Create structured events array
            num_events = len(spike_ids_current)
            events_new = np.column_stack([
                coords_2d[1], # x (width dimension)
                coords_2d[0], # y (height dimension)
                polarities,
                np.full(num_events, iteration * self.dt, dtype=np.int32)
            ])
            
            names = ["x", "y", "p", "t"]
            formats = ['i4', 'i4', 'i4', 'i4']
            dtype = np.dtype({'names': names, 'formats': formats})
            
            downsampled_events = unstructured_to_structured(events_new, dtype=dtype)
        else:
            # No spikes, return empty array
            names = ["x", "y", "p", "t"]
            formats = ['i4', 'i4', 'i4', 'i4']
            dtype = np.dtype({'names': names, 'formats': formats})
            downsampled_events = np.array([], dtype=dtype)
        
        return downsampled_events
    
    # def reset(self):
    #     """Reset the model state."""
    #     self.magnitude_view[:] = 0.0
    #     self.V_view[:] = 0.0
    #     self.Theta0_view[:] = 0.0
    #     self.Theta1_view[:] = 0.0
    #     self.polarity_view[:] = 0
        
    #     self.model.push_var_to_device("histogram_input", "magnitude")
    #     self.model.push_var_to_device("neurons", "V")
    #     self.model.push_var_to_device("neurons", "Theta0")
    #     self.model.push_var_to_device("neurons", "Theta1")
    #     self.model.push_var_to_device("neurons", "LastSpikePol")


# def event_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float,
#                      tau_theta: float, tau_accumulator: float, beta: float, iteration: int,
#                      genn_model_instance, **kwargs):
#     """
#     Wrapper function to maintain API compatibility with original implementation.
    
#     Parameters:
#         events (ndarray): ndarray of shape [num_events, num_event_channels].
#         sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
#         target_size (tuple): a 2-tuple of x,y denoting downsampled size.
#         dt (float): temporal resolution of events in milliseconds.
#         tau_theta (float): time constant of adaptive threshold (milliseconds).
#         tau_accumulator (float): time constant of accumulator (milliseconds).
#         beta (float): constant to scale threshold adaptation.
#         iteration (int): current iteration number.
#         genn_model_instance (EventDownsampleGeNN): the GeNN model instance.
        
#     Returns:
#         downsampled events.
#     """
#     if iteration == 0:
#         assert {"x", "y"} <= set(events.dtype.names)
        
#         # Normalise event key names
#         event_key_map = {"timestamp": "t", "polarity": "p"}
#         event_key_names = list(events.dtype.names)
        
#         for old_key, new_key in event_key_map.items():
#             if old_key in event_key_names:
#                 event_key_names[event_key_names.index(old_key)] = new_key
        
#         events.dtype.names = tuple(event_key_names)
    
#     return genn_model_instance.process_events(events, iteration)