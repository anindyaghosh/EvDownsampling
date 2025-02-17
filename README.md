# EvDownsampling

**Authors:** Anindya Ghosh, Thomas Nowotny, James Knight

EvDownsampling allows real-time **event-based spatio-temporal downsampling**. It has been tested at various spatial downsampling factors on the DVSGesture dataset with minimal reduction in classification accuracy even at high spatial downsampling factors, while significantly reducing the number of events needed to achieve a high accuracy by a downstream SNN.

We also tested EvDownsampling on its ability to successfully provide lower-resolution events from higher-resolution event streams by comparing downsampled event streams to actual lower-resolution event streams. For this we compared the downsampled event streams from a DVXplorer (640x480 pixels) to the event streams obtained from a Davis346 (346x260 pixels) DVS cameras — both event streams were of the same scene.

The event streams of the scenes for both the DXplorer and the Davis346 can be found in the [EvDownsampling dataset](https://doi.org/10.25377/sussex.26528146).

![Traffic gif](https://github.com/user-attachments/assets/97016855-6f7f-40a8-a3bb-21f5bb92748c)

### Related Publications
Ghosh, A., Nowotny, T. and Knight, J., 2024. **EvDownsampling: A Robust Method For Downsampling Event Camera Data**. In ECCV Workshop on Neuromorphic Vision: Advantages and Applications of Event Cameras. [PDF](https://drive.google.com/file/d/1s40YRb1HdJ7GMWotIpakDeKl9ETv8dd6/view).

Ghosh, A., Nowotny, T. and Knight, J.C., 2023, August. **Insect-inspired Spatio-temporal Downsampling of Event-based Input**. In Proceedings of the 2023 International Conference on Neuromorphic Systems (pp. 1-5). [PDF](https://dl.acm.org/doi/pdf/10.1145/3589737.3605994).

Ghosh, A., Nowotny, T. and Knight, J., 2022. **Event-based spatio-temporal down-sampling. In UKRAS22 Conference “Robotics for Unconstrained Environments"**. UKRAS22 Conference “Robotics for Unconstrained Environments" Proceedings (Vol. 5, pp. 26-27). [PDF](https://www.researchgate.net/profile/Anindya-Ghosh-14/publication/365398835_Event-based_Spatio-temporal_down-sampling/links/63ee13cb2958d64a5cd5b583/Event-based-Spatio-temporal-down-sampling.pdf).

## 1. License
```
@article{Ghosh2024,
author = "Anindya Ghosh and Thomas Nowotny and James Knight",
title = "{EvDownsampling: a robust method for downsampling event camera data}",
year = "2024",
month = "9",
url = "https://sussex.figshare.com/articles/conference_contribution/EvDownsampling_a_robust_method_for_downsampling_event_camera_data/26970640",
}
```

## 2. Prerequisite

To ensure rapid building of 2D histograms, we use Tonic's [ToFrame](https://tonic.readthedocs.io/en/latest/auto_examples/representations/plot_toframe.html#sphx-glr-auto-examples-representations-plot-toframe-py) class. As such, Tonic needs to be installed. It can be installed using:
```
pip install tonic
```

## 3. Running EvDownsampling and examples

### Directory structure

The EvDownsampling dataset should be on the same level as the Python scripts. An example of this is shown below:

```
EvDownsampling
├───event_based_downsampling.py
├───data
    ├───EvDownsampling_cars
    │   ├───aedat
    │   ├───numpy
    ├───EvDownsampling_corridor
    │   ├───aedat
    ├───EvDownsampling_handGestures
    │   ├───aedat
    ├───EvDownsampling_traffic
    │   ├───aedat
```

An example script of how to run EvDownsampling is shown in ```testing.py```. The example script:
  - Downsamples the ```Arundel``` event stream found in the ```corridor``` folder.
  - Visualises the downsampled event stream.

All the downsampling and visualisation parameters can be found and changed in ```EvDownsampling_config.yaml```.
