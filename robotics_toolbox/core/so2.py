#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from typing import Self
import numpy as np
from numpy.typing import ArrayLike


class SO2:
    """This class represents an SO2 rotations internally represented by rotation
    matrix."""

    def __init__(self, angle: float = 0.0, degrees=False) -> None:
        """Creates a rotation transformation that rotates vector by a given angle, that
        is expressed in radians unless variable degrees is set to true."""
        super().__init__()
        # todo HW01: implement computation of rotation matrix from the given angle
        self.rot: np.ndarray = np.zeros((2, 2))

    def __mul__(self, other: Self) -> Self:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        pass

    @property
    def angle(self) -> float:
        """Return angle from the internal rotation matrix representation."""
        # todo: HW01: implement computation of rotation matrix from angles.
        angle = 0.0
        return angle

    def transform(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given 2D vector by this SO2 transformation."""
        v = np.asarray(vector)
        assert v.shape == (2,)
        return self.rot @ v

    def __eq__(self, other: Self) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)