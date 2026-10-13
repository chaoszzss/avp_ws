# Assignment ⅤI: AVP Localization
  For this assignment, you are required to complete the localization algorithm for the AVP system. The assignment provides a dataset of segmented ipm images from simulation. You need to complete the process of the localization algorithm and display the final result of the AVP localization.  

  - Complete imageRegistration() in src/avp_localization.cpp  

## Build & Run

Terminal (Open in l6_ws): 

```bash 1
catkin_make 
source ./devel/setup.bash
roslaunch avp_localization avp_localization.launch
```
 - roslaunch will load data/avp_map.bin and play bag automatically.  
 - you can try  avp_map.bin you built in ***Assignment Ⅴ***

# Submission
You need to submit a ***PDF*** file that contains:  
- The completed code, with your explanation;
- The final plot the path of your avp localization;
- ***DO NOT*** submit the "l6_ws" and "avp_localization/data" dataset.  
