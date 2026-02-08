import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState, Imu
from math import cos, sin
import numpy as np

class GausianKF(Node):

    def __init__(self):
        super().__init__('kf_node')

        self.w, self.v = 0.0
        self.zt = []
        self.ut = []
        self.B = np.array([[0.0], [1.0]])
        self.F = []
        self.H = np.array([[0.0, 1.0]])

        ##subscribe from /imu
        self.create_subscription(JointState,  "/joint_states", self.process_sensor_data, 10 )
        self.create_subscription(Vector3, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)

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


    def cmd_callback(self, msg: Vector3):

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
        self.uk = [msg.linear.x, msg.angular.z]


def main():
    rclpy.init()

    node = GausianKF()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
