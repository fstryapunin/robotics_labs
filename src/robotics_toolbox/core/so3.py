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
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)

        angle = np.linalg.norm(v)
        axis = np.multiply(v, 1 / angle)
        scew_symmetric = SO3.to_skew_symmetric(axis)

        identity = np.identity(3)
        sinpart = np.multiply(-1 * np.sin(angle), scew_symmetric)
        cospart = np.multiply(1 - np.cos(angle), scew_symmetric @ scew_symmetric)

        return SO3(identity + sinpart + cospart)

    def log(self) -> np.ndarray | None:
        """Compute rotation vector from this SO3"""
        if(np.array_equal(np.identity(3), self.rot)):
            return np.zeros(3)
        
        trace = np.trace(self.rot)
        angle = np.arccos((1/2)*(trace - 1))
        vector = np.asarray([item * angle for item in SO3.from_skew_symmetric(np.multiply(1/(2 * np.sin(angle)), self.rot - self.rot.transpose()))])

        return vector
    

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
        # todo: HW1opt: implement rx
        raise NotImplementedError("RX needs to be implemented.")

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # todo: HW1opt: implement ry
        raise NotImplementedError("RY needs to be implemented.")

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # todo: HW1opt: implement rz
        raise NotImplementedError("RZ needs to be implemented.")

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternion
        raise NotImplementedError("From quaternion needs to be implemented")

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # todo: HW1opt: implement to quaternion
        raise NotImplementedError("To quaternion needs to be implemented")

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        raise NotImplementedError("Needs to be implemented")

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # todo: HW1opt: implement to angle axis
        raise NotImplementedError("Needs to be implemented")

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
