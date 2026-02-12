import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry, Path

from math import cos, sin
import numpy as np


class GaussianKF(Node):
    def __init__(self):
        super().__init__("kf_node")

       
    
        self.r = 0.033   # wheel radius
        self.b = 0.160   # wheel separation

        self.kf_path = Path()
        self.kf_path.header.frame_id = "odom"

        
        # State: x = [v, w]
        self.x = np.zeros((2, 1), dtype=float)
        self.P = np.eye(2, dtype=float) * 0.1

        # Process + measurement noise
        self.Q = np.diag([0.07**2, 0.01**2]).astype(float)

        self.R = np.diag([0.03**2, 0.15**2, 0.001**2]).astype(float)
        self.H = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=float)

        
        # i tried to treat command as a direct inputs for v,w
        self.F = np.eye(2, dtype=float)
        self.B = np.eye(2, dtype=float)

        # Latest measurements
        self.z = None              
        self.u = np.zeros((2, 1))  

        # Timing for filter + integration
        self.last_time = None

        self.theta = 0.0
        self.px = 0.0
        self.py = 0.0

        self.last_wheel_pos = None
        self.last_wheel_t = None

        
        # Subsscriptions and a publisher
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, "/kf_odom", 10)
        self.kf_path_pub = self.create_publisher(Path, "kf_path", 10)

        self.timer = self.create_timer(0.02, self.kf_step)

        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0

    
    def joint_callback(self, msg: JointState):

        names = ["wheel_left_joint", "wheel_right_joint"]
        try:
            wl_i = msg.name.index(names[0])
            wr_i = msg.name.index(names[1])
        except ValueError:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9
        wl_pos = float(msg.position[wl_i])
        wr_pos = float(msg.position[wr_i])

        if self.last_wheel_pos is None:
            self.last_wheel_pos = (wl_pos, wr_pos)
            self.last_wheel_t = t_now
            return

        dt = t_now - self.last_wheel_t
        if dt < 0.015:
            return

        wl_prev, wr_prev = self.last_wheel_pos
        wl = (wl_pos - wl_prev) / dt   # rad/s
        wr = (wr_pos - wr_prev) / dt   # rad/s

        self.last_wheel_pos = (wl_pos, wr_pos)
        self.last_wheel_t = t_now

        self.v_enc = (self.r / 2.0) * (wl + wr)
        self.w_enc = (self.r / self.b) * (wr - wl)

        self.w_enc = - self.w_enc

        self.update_z_vector()

    def update_z_vector(self):
        self.z = np.array([
            [self.v_enc],
            [self.w_enc],
            [self.w_imu]
        ], dtype=float)

    def imu_callback(self, msg: Imu):
        self.w_imu = float(msg.angular_velocity.z)
        if self.z is not None:
            self.z[2, 0] = self.w_imu

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([[v_cmd], [w_cmd]], dtype=float)


    def predict(self):       
        x_pred = self.F @ self.x + self.B @ self.u
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred):

        # x = x_pred + K y
        # S = H P_pred H^T + R
        # K = P_pred H^T S^-1
        # P = (I - K H) P_pred
        # y = z - H x_pred

        if self.z is None:
            self.x = x_pred
            self.P = P_pred
            return


        y = self.z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y)
        I = np.eye(2, dtype=float)
        self.P = (I - (K @ self.H)) @ P_pred

   
    def kf_step(self):

        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9
        current_time_msg = t_now.to_msg()
       
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        
        x_pred, P_pred = self.predict()
        self.update(x_pred, P_pred)

        v_f = float(self.x[0, 0])
        w_f = float(self.x[1, 0])

        # Integrate pose using filtered v,w
        self.theta += w_f * dt
        self.px += v_f * cos(self.theta) * dt
        self.py += v_f * sin(self.theta) * dt

        # Publish odom
        odom = Odometry()
        odom.header.stamp = current_time_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = self.px
        odom.pose.pose.position.y = self.py
        odom.pose.pose.position.z = 0.0

        odom.twist.twist.linear.x = v_f
        odom.twist.twist.angular.z = w_f

        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header.stamp = current_time_msg
        pose.header.frame_id = "odom"
        pose.pose = odom.pose.pose
        
        self.kf_path.poses.append(pose)
        self.kf_path.header.stamp = current_time_msg
        self.kf_path_pub.publish(self.kf_path)






def main():
    rclpy.init()
    node = GaussianKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
