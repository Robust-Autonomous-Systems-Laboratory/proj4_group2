# proj4_group2

## Data Collection Using Turtlebot 3

["Video Controlling TurtleBot"](https://drive.google.com/file/d/1w8894PAhPiHxX0CNOtRNaqFdzJPt3LTD/view?usp=drive_link)

## How to run

## 1. Step 1 creating workspace

```bash
mkdir -p ~/project_2/
cd ~/project_2/
```

## 2.  Clone the repo from github

```bash
git clone https://github.com/Robust-Autonomous-Systems-Laboratory/proj4_group2.git
```
## 3. Build and source the package

```bash
colcon build --symlink-install
source install/setup.bash
```
## 4. Run the Node

```bash
ros2 run  gaussian_filters {filter_node_name}
```
# for filter node names

- kf_node - *linear kalman filter*
- ekf_node - *extended kalman filter*
- ukf_node - *uncented kalman filter*

## 5. Assuming lab4_ws is availabe (open new terminal)

```bash
source install/setup.bash
ros2 launch turtlebot3_bringup rviz2.launch.py
```

this should open rviz and the  turtlebot model

## 6. In RViz
- change the odometry -> topic to either /kf_odom or ekf_odom or ukf_odom
![odometry Topic Change](./images/image1.png)
- Add path and change the topic to either /kf_path or ekf_path or ukf_path
![Path Topic Change](./images/image2.png)

## 7 Play the bag file (open new terminal)
```bash
ros2 bag play rosbag2_final/
```


