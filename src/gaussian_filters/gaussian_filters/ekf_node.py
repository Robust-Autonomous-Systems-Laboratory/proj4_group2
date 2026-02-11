import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped, TransformStamped, PoseStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster


from math import cos, sin
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi



def to_quaternion(agnle: float):
    qz = np.sin(agnle * 0.5)
    qw = np.cos(agnle * 0.5)
    return (0.0, 0.0, qz, qw)


class GaussianEKF(Node):
    
    def __init__(self):
        super().__init__("ekf_node")



        self.ekf_path = Path()
        self.ekf_path.header.frame_id = "odom"

        
        self.r = 0.033   
        self.b = 0.160

        
        self.x = np.zeros((5, 1), dtype=float)
        self.P = np.eye(5, dtype=float) * 0.1

        # Process noise
        # px, py, theta, v, w
        self.Q = np.diag([0.01**2, 0.01**2, 0.02**2, 0.20**2, 0.30**2]).astype(float)

        # Measrment noise for [v_enc, w_enc, w_imu]
        self.R = np.diag([0.03**2, 0.08**2, 0.05**2]).astype(float)

        self.z = None
        self.u = np.zeros((2, 1), dtype=float)

        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0

        # Timing
        self.last_time = None

        self.alpha_v = 8.0
        self.alpha_w = 10.0

        self.tf_broadcaster = TransformBroadcaster(self)

        
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, "/ekf_odom", 10)
        self.ekf_path_pub = self.create_publisher(Path, "ekf_path", 10)
        self.timer = self.create_timer(0.02, self.ekf_step)

   
    def joint_callback(self, msg: JointState):
        names = ["wheel_left_joint", "wheel_right_joint"]
        try:
            wl_i = msg.name.index(names[0])
            wr_i = msg.name.index(names[1])
        except ValueError:
            return

        wl = float(msg.velocity[wl_i])
        wr = float(msg.velocity[wr_i])

        self.v_enc = (self.r / 2.0) * (wl + wr)
        self.w_enc = (self.r / self.b) * (wl - wr)

        self.z = np.array([[self.v_enc], [self.w_enc], [self.w_imu]], dtype=float)

    def imu_callback(self, msg: Imu):
        self.w_imu = float(msg.angular_velocity.z)

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([[v_cmd], [w_cmd]], dtype=float)

    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        px, py, th, v, w = x.flatten()
        v_cmd, w_cmd = u.flatten()

        px_new = px + v * dt * cos(th)
        py_new = py + v * dt * sin(th)
        th_new = wrap_angle(th + w * dt)

        v_new = v + self.alpha_v * (v_cmd - v) * dt
        w_new = w + self.alpha_w * (w_cmd - w) * dt

        return np.array([[px_new], [py_new], [th_new], [v_new], [w_new]], dtype=float)
    

    def jacobian_f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
       
        _, _, th, v, w = x.flatten()
        F = np.eye(5, dtype=float)
        F[0, 2] = -v * dt * sin(th)
        F[0, 3] =  dt * cos(th)
        F[1, 2] =  v * dt * cos(th)
        F[1, 3] =  dt * sin(th)
        F[2, 4] = dt
        F[3, 3] = 1.0 - self.alpha_v * dt
        F[4, 4] = 1.0 - self.alpha_w * dt
        return F

    def h(self, x: np.ndarray) -> np.ndarray:
       
        v = float(x[3, 0])
        w = float(x[4, 0])
        return np.array([[v], [w], [w]], dtype=float)

    def jacobian_h(self, x: np.ndarray) -> np.ndarray:
       
        H = np.zeros((3, 5), dtype=float)
        H[0, 3] = 1.0
        H[1, 4] = 1.0  
        H[2, 4] = 1.0 
        return H

    
    def predict(self, dt: float):
        x_pred = self.f(self.x, self.u, dt)
        F = self.jacobian_f(self.x, self.u, dt)
        P_pred = F @ self.P @ F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray):
        if self.z is None:
            self.x = x_pred
            self.P = P_pred
            self.x[2, 0] = wrap_angle(self.x[2, 0])
            return

        z_pred = self.h(x_pred)
        H = self.jacobian_h(x_pred)
        y = self.z - z_pred
        S = H @ P_pred @ H.T + self.R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros_like(P_pred @ H.T)

        self.x = x_pred + (K @ y)
        self.x[2, 0] = wrap_angle(self.x[2, 0])
        I = np.eye(5, dtype=float)
        self.P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ self.R @ K.T

    
    def ekf_step(self):
       
        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        # EKF
        x_pred, P_pred = self.predict(dt)
        self.update(x_pred, P_pred)

        # Publish odom from EKF pose and filtered v,w
        px = float(self.x[0, 0])
        py = float(self.x[1, 0])
        theta = float(self.x[2, 0])
        v_f = float(self.x[3, 0])
        w_f = float(self.x[4, 0])

        qx, qy, qz, qw = to_quaternion(theta)

        current_time_msg = t_now.to_msg()

        
        odom = Odometry()
        odom.header.stamp = current_time_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = 0.0
        
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw


        odom.twist.twist.linear.x = v_f
        odom.twist.twist.angular.z = w_f
        self.odom_pub.publish(odom)


        pose = PoseStamped()
        pose.header.stamp = current_time_msg
        pose.header.frame_id = "odom"
        pose.pose = odom.pose.pose
        
        self.ekf_path.poses.append(pose)
        self.ekf_path.header.stamp = current_time_msg
        self.ekf_path_pub.publish(self.ekf_path)


def main():
    rclpy.init()
    node = GaussianEKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
