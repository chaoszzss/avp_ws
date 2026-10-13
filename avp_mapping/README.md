# Assignment Ⅴ: AVP  Mapping
  For this assignment, you are required to complete the mapping algorithm for the AVP system. The assignment provides a dataset of segmented ipm images from simulation. You need to complete the process of the mapping algorithm and display the final result of the AVP mapping.  

- hw1: Complete ipmPlane2Global()  in avp_mapping.cc    
- hw2: Complete detectSlot() in slot.cc     
- hw3: Explain the effects of image-processing operations in extractSlot(), adjust their parameters and try to design your image processing method, is your result better?  

## Build & Run

Terminal (Open in l5_ws): 

```bash
catkin_make 
source ./devel/setup.bash
roslaunch avp_mapping avp_mapping.launch
```
 
There is "ipm_detection_result" window when running, showing the detection result of current ipm image;   
- press 's' to enter "step by step" mode, program will wait for keyboard input before handling next ipm image;    
- press 'r' to enter Automatic mode, program will run automatically;  
- The default mode is Automatic.  

You can see the result of avp mapping in rviz.

# Semantic vs. Geometric Heatmap Outputs

After running `roslaunch avp_mapping avp_mapping.launch`, the node now serializes the
semantic map (`map_save_dir/avp_map_sim.bin`) and exports a semantic coverage heatmap
(`semantic_heatmap_path`, defaults to `map_save_dir/semantic_coverage_heatmap.png`).
Configure the paths in the launch file or with ROS parameters:

```bash
roslaunch avp_mapping avp_mapping.launch map_save_dir:=/tmp/avp_maps \
  semantic_heatmap_path:=/tmp/avp_maps/semantic_heatmap.png
```

Semantic elements are downsampled before serialization. Adjust the voxel size via
`semantic_voxel_size` (meters, default `0.1`) if needed:

```bash
roslaunch avp_mapping avp_mapping.launch semantic_voxel_size:=0.1
```

The geometric mapping node saves the accumulated point cloud map (`map_save_path`) and
a geometric density heatmap (`density_heatmap_path`):

```bash
roslaunch avp_mapping geometric_mapping.launch \
  map_save_path:=/tmp/geom_map.pcd \
  density_heatmap_path:=/tmp/geom_density.png
```

For offline comparison, use `map_heatmap_tool` once both maps are produced:

```bash
rosrun avp_mapping map_heatmap_tool \
  --semantic_map /tmp/avp_maps/avp_map_sim.bin \
  --geometric_map /tmp/geom_map.pcd \
  --output_dir /tmp/heatmaps --resolution 0.5
```

To evaluate coverage metrics:

```bash
rosrun avp_mapping coverage_metrics_tool \
  --semantic_map /tmp/avp_maps/avp_map_sim.bin \
  --geometric_map /tmp/geom_map.pcd \
  --output_dir /tmp/Quantitative_indicators --resolution 0.2
```

The tool dumps per-category masks (PNG) and `coverage_metrics.csv` with
semantic area, reference geometric area, and the resulting coverage ratio.

To compare data volume and storage:

```bash
rosrun avp_mapping storage_metrics_tool \
  --semantic_map /tmp/avp_maps/avp_map_sim.bin \
  --geometric_map /tmp/geom_map.pcd \
  --output_dir /tmp/Quantitative_indicators/save_num
```

Outputs `storage_metrics.csv` summarizing element counts, file sizes, density
per m², and the semantic-to-geometric size ratio.

To collect entropy/diversity indicators and export CSV/heatmaps:

```bash

# Submission
You need to submit a ***PDF*** file that contains:  
- The completed code of hw1 and hw2, with explanation of your code;
- The report of hw3;
- The result(final rviz image) of your avp mapping.
- ***DO NOT*** submit the "l5_ws" and "ipm_data" dataset.  
