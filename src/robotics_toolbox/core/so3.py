#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def rodriguez(axis, angle) -> np.ndarray:
        scew_symmetric = SO3.to_skew_symmetric(axis)

        identity = np.identity(3)
        sinpart = np.multiply(-1 * np.sin(angle), scew_symmetric)
        cospart = np.multiply(1 - np.cos(angle), scew_symmetric @ scew_symmetric)

        return identity + sinpart + cospart

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)

        angle = np.linalg.norm(v)
        axis = np.multiply(v, 1 / angle)

        return SO3(SO3.rodriguez(axis, angle))
    
    def axis_from_rot(self) -> np.array:
        if(np.array_equal(np.identity(3), self.rot)):
            return np.zeros(3)
        
        angle = self.angle_from_rot()

        if(angle == np.pi):
            print(self.rot)
            if self.rot[2][2] != -1:
                return np.multiply((1 / np.sqrt(2 * (self.rot[2][2] + 1))), [self.rot[0][2], self.rot[1][2], self.rot[2][2] + 1]) * angle
            if self.rot[1][1] != -1:
                return np.multiply((1 / np.sqrt(2 * (self.rot[1][1] + 1))), [self.rot[0][1], self.rot[1][1] + 1, self.rot[2][1]]) * angle
            if self.rot[0][0] != -1:
                return np.multiply((1 / np.sqrt(2 * (self.rot[0][0] + 1))), [self.rot[0][0] + 1, self.rot[1][0], self.rot[2][0]]) * angle
                        
        vector = np.asarray([item * angle for item in SO3.from_skew_symmetric(np.multiply(1/(2 * np.sin(angle)), self.rot - self.rot.transpose()))])
        
        return vector
    
    def angle_from_rot(self) -> float:
        if(np.array_equal(np.identity(3), self.rot)):
            return 0
        
        trace = np.trace(self.rot)

        if(trace == -1):
            return np.pi
        
        return np.arccos((1/2)*(trace - 1))

    def log(self) -> np.ndarray | None:
        """Compute rotation vector from this SO3"""
        if(np.array_equal(np.identity(3), self.rot)):
            return np.zeros(3)

        return self.axis_from_rot()
    

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        return SO3(self.rot @ other.rot)

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        return SO3(self.rot.transpose())

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        return SO3([[1, 0, 0], [0, np.cos(angle), -1 * np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        return SO3([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-1 * np.sin(angle), 0, np.cos(angle)]])

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        return SO3([[np.cos(angle), -1 * np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        xyz_vector = q[:3]
        xyz_axis = np.multiply(xyz_vector, 1 / np.linalg.norm(xyz_vector))
        angle = 2 * np.arccos(q[3])
        return SO3(SO3.rodriguez(xyz_axis, angle))

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        w = 1/2 * np.sqrt(1 + np.trace(self.rot))
        w_multiplier = 1 / (4 * w)
        x = (self.rot[2][1] - self.rot[1][2]) * w_multiplier
        y = (self.rot[0][2] - self.rot[2][0]) * w_multiplier
        z = (self.rot[1][0] - self.rot[0][1]) * w_multiplier
        return np.asarray([x, y, z, w])

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        return SO3(SO3.rodriguez(axis, angle))

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        return (self.angle_from_rot(), self.axis_from_rot() / self.angle_from_rot())

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        raise NotImplementedError("Needs to be implemented")
    
    @staticmethod
    def to_skew_symmetric(vector: ArrayLike) -> np.ndarray:
        v_x = vector[0]
        v_y = vector[1]
        v_z = vector[2]
        return np.asarray([[0, v_z, -1 * v_y], [-1 * v_z, 0, v_x], [v_y, -1 * v_x, 0]])
    
    @staticmethod
    def from_skew_symmetric(matrix: ArrayLike) -> np.array:
        return np.asarray([matrix[2][1], matrix[0][2], matrix[1][0]])


    def __hash__(self):
        return id(self)
    
    def __str__(self) -> str:
        np.set_printoptions(precision=3)
        return print(self.rot)
    
    def __repr__(self) -> str:
        self.__str__()
