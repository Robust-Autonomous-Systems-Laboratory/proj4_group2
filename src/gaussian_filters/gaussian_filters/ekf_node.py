import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry

from math import cos, sin
import numpy as np


class GaussianEKF(Node):
    def __init__(self):
        super().__init__("ekf_node")

       
    
        self.r = 0.033   # wheel radius (m)
        self.b = 0.160   # wheel separation (m)

        
        # State: x = [v, w]^T
        
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 0.1

        # Process + measurement noise

        # Q: uncertainty in how v,w evolve between steps
        self.Q = np.diag([0.05**2, 0.10**2]).astype(float)

        self.R = np.diag([0.03**2, 0.08**2, 0.05**2]).astype(float)

        # Measurement model: z = Hx + noise
        # v_enc measures v
        # w_enc measures w
        # w_imu measures w
        self.H = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=float)

        
        # i tried to reat command as a direct inputs for v,w
        self.F = np.eye(2, dtype=float)
        
        # Latest measurements
        self.z = None              
        self.u = np.zeros((2, 1))  

        # Timing for filter + integration
        self.last_time = None

        # Pose (integrated from filtered v,w)
        self.theta = 0.0
        self.px = 0.0
        self.py = 0.0

        
        # Subsscriptions and a publisher

        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, "/ekf_odom", 10)

        self.timer = self.create_timer(0.02, self.kf_step)

        # Cache encoder and imu values
        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0

    
    def joint_callback(self, msg: JointState):

        #Using the joint names so that we dont assume what the index of left or right wheel are

        names = ["wheel_left_joint", "wheel_right_joint"]
        try:
            wl_i = msg.name.index(names[0])
            wr_i = msg.name.index(names[1])

        except ValueError:
            return

        wl = msg.velocity[wl_i]
        wr = msg.velocity[wr_i]

        self.v_enc = (self.r / 2.0) * (wl + wr)
        self.w_enc = (self.r / self.b) * (wr - wl)

        
        self.z = np.array([[self.v_enc], [self.w_enc], [self.w_imu]], dtype=float)

    def imu_callback(self, msg: Imu):
        self.w_imu = float(msg.angular_velocity.z)
        if self.z is not None:
            self.z[2, 0] = self.w_imu

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([[v_cmd], [w_cmd]], dtype=float)


    def func_x(self, dt):

        u = self.u.flatten()
        x = self.x.flatten()

        x_new = u[0]*dt*cos(x[2]) + x[0]
        y_new = u[0]*dt*cos(x[2]) + x[1]
        theta_new = x[2] + u[1]*dt
        v_new = u[0]

        return np.array([
            [x_new],[y_new], [theta_new], [v_new]

        ])
    
    def jacobian_f(self,fx, dt, eps=1e-6):

        n = self.x.shape[0]
        F = np.zeros((n, n))
 
        for i in range(n):
            dx = np.zeros_like(self.x)
            dx[i, 0] = eps
            F[:, i:i+1] = (self.func_x(self.x + dx, self.u, dt) - fx) / eps

        return F
    
    def numerical_jacobian_h(self, func_h, eps=1e-6):
        z0 = func_h(self.x)
        m = z0.shape[0]
        n = self.x.shape[0]

        H = np.zeros((m, n))

        for i in range(n):
            dx = np.zeros_like(self.x)
            dx[i, 0] = eps
            H[:, i:i+1] = (func_h(self.x + dx) - z0) / eps

        return H
    
    def h(x):
        return x[0:2]
    

    def jacobian_h(self,x,eps=1e-6):

        z0 = self.h(x)
        m = z0.shape[0]
        n = x.shape[0]

        H = np.zeros((m, n))

        for i in range(n):
            dx = np.zeros_like(x)
            dx[i, 0] = eps
            H[:, i:i+1] = (self.h(x + dx) - z0) / eps

        return H


    def predict(self, dt): 

        x_pred = self.func_x(dt)
        self.F = self.jacobian_f(x_pred, dt)
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred):

        # x = x_pred + K y
        # S = H P_pred H^T + R
        # K = P_pred H^T S^-1
        # P = (I - K H) P_pred
        # y = z - H x_pred

        z_pred = self.h(x_pred)
        y = self.z - z_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y)
        I = np.eye(2, dtype=float)
        self.P = (I - (K @ self.H)) @ P_pred

   
    def ekf_step(self):

        if self.z is None:
            return

        # dt
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        
        x_pred, P_pred = self.predict(dt)
        self.update(x_pred, P_pred)

        v_f = float(self.x[0, 0])
        w_f = float(self.x[1, 0])

        # Integrate pose using filtered v,w
        self.theta += w_f * dt
        self.px += v_f * cos(self.theta) * dt
        self.py += v_f * sin(self.theta) * dt

        # Publish odom
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.px
        odom.pose.pose.position.y = self.py
        odom.pose.pose.position.z = 0.0

        odom.twist.twist.linear.x = v_f
        odom.twist.twist.angular.z = w_f

        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = GaussianEKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
