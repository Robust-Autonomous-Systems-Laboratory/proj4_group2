import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry, Path
from math import cos, sin
import numpy as np

class GausianKF(Node):

    def __init__(self):
        super().__init__('kf_node')

        self.w, self.v = 0.0
        self.x_vw = np.zeros((2,1), dtype=float)
        
        self.zt = []
        self.ut = []

        self.B = np.array([[0.0], [1.0]])
        self.F = []
        self.H = np.array([[0.0, 1.0]])
        self.P = np.eye(2, dtype=float) * 0.1

        self.R = np.diag([0.03 **2, 0.08 **2, 0.05 **2]).astype(float)
        self.Q = np.diag([0.05 **2 , 0.10 **2]).astype(float)

        self.theta = 0.0
        self.px = 0.0
        self.py = 0.0




    

        ##subscribe from /imu
        self.create_subscription(JointState,  "/joint_states", self.process_sensor_data, 10 )
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)

        #Publisher
        self.odom_pub = self.create_publisher(Odometry, "/KF_odom", 10)


    def get_dt(self, msg, last_time):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if last_time is None:
            return 0.0, t
        return t - last_time, t
    

    def process_sensor_data(self, msg: JointState):

        w_r = 0.033
        w_b = 0.160

        names = ["wheel_left_joint", "wheel_right_joint"]

        wl_index = msg.name.index(names[0])
        wr_index = msg.name.index(names[1])

        wl_v = msg.velocity[wl_index]
        wr_v = msg.velocity[wr_index]

        self.v = w_r/2 * (wl_v + wr_v)
        self.w = w_r / w_b * (wr_v - wl_v)

    

    def imu_callback(self, msg: Imu):

        wz = msg.angular_velocity.z
        self.zk = [self.v, self.w, wz]


    def cmd_callback(self, msg: TwistStamped):

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_cmd_time is None:
            self.last_cmd_time = t_now
            return
        dt = t_now - self.last_cmd_time

        self.F = np.array(
            [
                [1, dt],
                [0, 1],
            ]
        )
        self.uk = [msg.twist.linear.x, msg.twist.angular.z]


    def propagate(self, F:np.array, P:np.array, Q: np.array):

        x_pred = F@self.x_vw + self.B@self.uk
        p_pred = P@F@F.T + Q

        return x_pred, p_pred
    
    def update(self, x_pred,p_pred):

        y = self.zk - self.H@ x_pred
        S = self.H@p_pred @self.H.T + self.R
        K = p_pred@self.H.T @ np.linalg.inv(S)

        I = np.eye(2, dtype=float)

        self.x_vw = x_pred + K@y
        self.P = (I - K@self.H)@p_pred

    def kalman_filter(self):

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_cmd_time is None:
            self.last_cmd_time = t_now
            return
        dt = t_now - self.last_cmd_time

        x_pred, p_pred = self.propagate(self.F, self.P, self.Q)
        
        self.update(x_pred, p_pred)

        v_f = float(self.x_vw[0,0])
        w_f = float(self.x_vw[0,1])


        #Integrating pose outside KF

        self.theta = w_f * dt
        self.px += v_f * cos(self.theta) * dt
        self.py += v_f * sin(self.theta) * dt


        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "Odom"
        odom.child_frame_id = "base_link"


        odom.pose.pose.position.x = self.px
        odom.pose.pose.position.y = self.py
        odom.pose.pose.position.z = 0.0

        odom.twist.twist.linear.x = v_f
        odom.twist.twist.angular.z = w_f


        self.odom_pub.publish(odom) 


def main():
    rclpy.init()

    node = GausianKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
