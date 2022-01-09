# KalmanFilter
# Author: Du Ang
# Based on https://github.com/dougszumski/KalmanFilter/blob/master/kalman_filter.py by Doug Szumski.
# Differences with the original version:
#   - add control term
#   - use numpy multiplication
# Materials references: http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# Date: July 1, 2018

import numpy as np
from numpy.linalg import inv
import random

class KalmanFilter:
    """
    Simple Kalman filter
    """

    def __init__(self, X,P, A, Q, Z, H, R, B=np.zeros(4), U=np.zeros(4),AT = 1):
        """
        Initialise the filter
        Args:
            X: State estimate
            A: Transition Matrix
            P: Estimate covariance
            B: Control matrix
            U: Control vector
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        w1 = random.randint(0,1)
        w2 = random.randint(0,1)
        w3 = random.randint(0,1)
        w4 = random.randint(0,1)

        W = np.array([[w1], [w2], [w3], [w4]], np.float32)

        self.X = X
        self.P = P
        self.A = A
        self.B = B
        self.U = U
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R
        self.AT = AT
        self.W = W

    def predict(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.M: Control vector
        Returns:
            updated self.X
        """
        w1 = random.randint(0,1)
        w2 = random.randint(0,1)
        w3 = random.randint(0,1)
        w4 = random.randint(0,1)

        W = np.array([[w1], [w2], [w3], [w4]], np.float32)
        self.W = W


        # Project the state ahead
        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.X
    def get_predict(self):
        return self.X
    def predict2(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.M: Control vector
        Returns:
            updated self.X
        """

        w1 = random.randint(0,1)
        w2 = random.randint(0,1)
        w3 = random.randint(0,1)
        w4 = random.randint(0,1)

        W = np.array([[w1], [w2], [w3], [w4]], np.float32)
        self.W = W

        transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.A = transitionMatrix
        # Project the state ahead
        if 1 :
            self.Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0,0.1]], np.float32)

        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ self.P @ self.A.T + self.Q



        return self.X

    def getprediction(self):
        return self.X

    def correct(self, Z):

        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
         
         => (4, 1)
         => (4, 4)
         => (4, 1)
         => (4, 4)
         => (4, 4)


        print("X shape", self.X.shape)
        print("A shape", self.A.shape)
        print("z shape", self.Z.shape)
        print("H shape", self.H.shape)
        print("R shape", self.R.shape)
        print("W shape", self.W.shape)
        print("K shape", K.shape)

        """

        #self.R = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)





        self.X += K @ (Z - self.H @ self.X)


        self.P = self.P - K @ self.H @ self.P



        return self.X

    def correctlinked(self, Z,H):

        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
        """
        self.H = H
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)


        self.X += K @ (Z - self.H @ self.X)


        self.P = self.P - K @ self.H @ self.P

        return self.X