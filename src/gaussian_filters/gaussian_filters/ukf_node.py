import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry

from math import cos, sin
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_quaternion(agnle: float):
    qz = np.sin(agnle * 0.5)
    qw = np.cos(agnle * 0.5)
    return (0.0, 0.0, qz, qw)


class GaussianUKF(Node):
   
    def __init__(self):
        super().__init__("ukf_node")

        # Robot params (Turtlebot-ish)
        self.r = 0.033 
        self.b = 0.160

        
        self.x = np.zeros((5, 1), dtype=float)
        self.P = np.eye(5, dtype=float) * 0.1


        #I estimated these values but i can later tune them as i test th ecode 
        self.Q = np.diag([0.01**2, 0.01**2, 0.02**2, 0.20**2, 0.30**2]).astype(float)
        self.R = np.diag([0.03**2, 0.08**2, 0.05**2]).astype(float)

        self.z = None
        self.u = np.zeros((2, 1), dtype=float)

        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0

        self.last_time = None


        self.alpha_v = 8.0
        self.alpha_w = 10.0

        self.n = 5
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.c = self.n + self.lam

        # Weights
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c), dtype=float)
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c), dtype=float)
        self.Wm[0] = self.lam / self.c
        self.Wc[0] = self.lam / self.c + (1.0 - self.alpha**2 + self.beta)

       
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, "/ukf_odom", 10)

        self.timer = self.create_timer(0.02, self.ukf_step)

   
    def joint_callback(self, msg: JointState):

        # Use joint names (no index assumptions)
        names = ["wheel_left_joint", "wheel_right_joint"]
        try:
            wl_i = msg.name.index(names[0])
            wr_i = msg.name.index(names[1])
        except ValueError:
            return

        wl = float(msg.velocity[wl_i])
        wr = float(msg.velocity[wr_i])

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

    
    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:

        """
        Motion model i decided to use the Forward kinematic Model as follows:
          px'    = px + v*dt*cos(theta)
          py'    = py + v*dt*sin(theta)
          theta' = theta + w*dt
          v'     = v + alpha_v*(v_cmd - v)*dt
          w'     = w + alpha_w*(w_cmd - w)*dt
        """


        px, py, th, v, w = x.flatten()
        v_cmd, w_cmd = u.flatten()

        px_new = px + v * dt * cos(th)
        py_new = py + v * dt * sin(th)
        th_new = wrap_angle(th + w * dt)

        v_new = v + self.alpha_v * (v_cmd - v) * dt
        w_new = w + self.alpha_w * (w_cmd - w) * dt

        return np.array([px_new, py_new, th_new, v_new, w_new], dtype=float)

    def h(self, x: np.ndarray) -> np.ndarray:

        
        v = float(x[3])
        w = float(x[4])
        return np.array([v, w, w], dtype=float)

   
    def sigma_points(self, mu: np.ndarray, P: np.ndarray) -> np.ndarray:

        #This part i had help its not all completely my code 

        """
        mu: (n,) mean
        P:  (n,n) cov
        return X: (2n+1, n)
        """
        # numerical safety
        P = 0.5 * (P + P.T)

        # Cholesky of (c * P)
        # add tiny jitter if needed
        jitter = 1e-12
        for _ in range(5):
            try:
                S = np.linalg.cholesky(self.c * (P + jitter * np.eye(self.n)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            # last resort: eigen fix
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, 1e-12, None)
            P = eigvecs @ np.diag(eigvals) @ eigvecs.T
            S = np.linalg.cholesky(self.c * P)

        X = np.zeros((2 * self.n + 1, self.n), dtype=float)
        X[0] = mu
        for i in range(self.n):
            col = S[:, i]
            X[i + 1] = mu + col
            X[i + 1 + self.n] = mu - col

        # sigma point angles should be wrapped
        X[:, 2] = np.vectorize(wrap_angle)(X[:, 2])
        return X

    def mean_and_cov(self, X: np.ndarray, mu_hint=None) -> tuple[np.ndarray, np.ndarray]:
        
        #This part i had help its not all completely my code also i was having problems here 

        # linear mean first
        mu = np.sum(self.Wm[:, None] * X, axis=0)

        # circular mean for theta
        sin_sum = np.sum(self.Wm * np.sin(X[:, 2]))
        cos_sum = np.sum(self.Wm * np.cos(X[:, 2]))
        mu[2] = np.arctan2(sin_sum, cos_sum)

        # covariance
        P = np.zeros((self.n, self.n), dtype=float)
        for i in range(X.shape[0]):
            d = X[i] - mu
            d[2] = wrap_angle(d[2])
            P += self.Wc[i] * np.outer(d, d)

        return mu, P

   
    def ukf_predict(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        mu = self.x.flatten()
        X = self.sigma_points(mu, self.P)

        # propagate sigma points
        X_pred = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_pred[i] = self.f(X[i], self.u, dt)

        mu_pred, P_pred = self.mean_and_cov(X_pred)
        P_pred += self.Q

        return X_pred, mu_pred, P_pred

    def ukf_update(self, X_pred: np.ndarray, mu_pred: np.ndarray, P_pred: np.ndarray):

        if self.z is None:
            self.x = mu_pred.reshape(-1, 1)
            self.P = P_pred
            return

        m = 3
        Z = np.zeros((2 * self.n + 1, m), dtype=float)
        for i in range(Z.shape[0]):
            Z[i] = self.h(X_pred[i])

        # predictd mesurement mean
        z_hat = np.sum(self.Wm[:, None] * Z, axis=0)

        # measurement covariance S
        S = np.zeros((m, m), dtype=float)
        for i in range(Z.shape[0]):
            dz = (Z[i] - z_hat).reshape(-1, 1)
            S += self.Wc[i] * (dz @ dz.T)
        S += self.R

        # cross covariance Pxz
        Pxz = np.zeros((self.n, m), dtype=float)
        for i in range(Z.shape[0]):
            dx = (X_pred[i] - mu_pred).reshape(-1, 1)
            dx[2, 0] = wrap_angle(dx[2, 0])
            dz = (Z[i] - z_hat).reshape(-1, 1)
            Pxz += self.Wc[i] * (dx @ dz.T)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # innovation
        y = (self.z.flatten() - z_hat).reshape(-1, 1)

        # update
        mu_new = mu_pred.reshape(-1, 1) + K @ y
        mu_new[2, 0] = wrap_angle(mu_new[2, 0])

        P_new = P_pred - K @ S @ K.T
        P_new = 0.5 * (P_new + P_new.T)  # symmetrize

        self.x = mu_new
        self.P = P_new

    
    def ukf_step(self):
        if self.z is None:
            return

        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        # UKF predict + update
        X_pred, mu_pred, P_pred = self.ukf_predict(dt)
        self.ukf_update(X_pred, mu_pred, P_pred)

        # Publish odom from UKF pose and filtered v,w
        px = float(self.x[0, 0])
        py = float(self.x[1, 0])
        theta = float(self.x[2, 0])
        v_f = float(self.x[3, 0])
        w_f = float(self.x[4, 0])

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = 0.0


        qx, qy, qz, qw = to_quaternion(theta)

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


def main():
    rclpy.init()
    node = GaussianUKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
