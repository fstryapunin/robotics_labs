#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 2D transformation."""

from __future__ import annotations

import copy
import numpy as np
from numpy.typing import ArrayLike

from robotics_toolbox.core import SO2


class SE2:
    """Transformation in 2D that is composed of rotation and translation."""

    def __init__(
        self, translation: ArrayLike | None = None, rotation: SO2 | float | None = None
    ) -> None:
        """Crete an SE2 transformation. Identity is the default. The se3 instance
        is represented by translation and rotation, where rotation is SO2 instance.

        Attributes:
        :param translation: 2D vector representing translation
        :param rotation: SO2 rotation or angle in radians that is converted to SO2
        """
        super().__init__()
        self.translation = (
            np.asarray(translation) if translation is not None else np.zeros(2)
        )
        assert self.translation.shape == (2,)
        if isinstance(rotation, SO2):
            self.rotation = rotation
        elif isinstance(rotation, float):
            self.rotation = SO2(rotation)
        else:
            self.rotation = SO2()

    def __mul__(self, other: SE2) -> SE2:
        """Compose two transformation, i.e., self * other"""
        return SE2(np.add(self.rotation.rot @ other.translation, self.translation), self.rotation * other.rotation)

    def inverse(self) -> SE2:
        """Compute inverse of the transformation. Do not use np.linalg.inv."""
        inverseRotation = self.rotation.inverse()
        return SE2(np.multiply(inverseRotation.rot @ self.translation, -1), inverseRotation)

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Transform given 2D vector by this SE2 transformation."""
        v = np.asarray(vector)
        assert v.shape == (2,)
        return np.add(self.rotation.rot @ vector, self.translation)

    def set_from(self, other: SE2):
        """Copy the properties into current instance."""
        self.translation = copy.deepcopy(other.translation)
        self.rotation = copy.deepcopy(other.rotation)

    def homogeneous(self) -> np.ndarray:
        """Return homogeneous transformation matrix."""
        h = np.eye(3)
        h[:2, :2] = self.rotation.rot
        h[:2, 2] = self.translation
        return h
    
    @staticmethod
    def from_homogenous(homogenous: np.ndarray) -> SE2:
        assert homogenous.shape == (3, 3)
        translation = [homogenous[0][2], homogenous[1][2]]
        rotation = SO2(np.array([[homogenous[0][0], homogenous[0][1]], [homogenous[1][0], homogenous[1][1]]]))
        return SE2(translation, rotation)

    def __eq__(self, other: SE2) -> bool:
        """Returns true if two transformations are almost equal."""
        return (
            np.allclose(self.translation, other.translation)
            and self.rotation == other.rotation
        )

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return (
            f"SE2(translation={self.translation}, rotation=SO2({self.rotation.angle}))"
        )
