#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
from functools import reduce
from typing import Callable, Union
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString
from copy import deepcopy

from robotics_toolbox.core import SE2, SE3, SO2
from robotics_toolbox.robots.robot_base import RobotBase

class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    @property
    def joint_count(self) -> int:
        return len(self.structure)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        transformations = [self.get_transformation_from_joint(joint_index) for joint_index in range(self.joint_count)]
        return self.__reduce_transformations(transformations, self.base_pose)

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        frames = [self.base_pose]

        for frame_index in range(self.joint_count):
            transformations = [self.get_transformation_from_joint(joint_index) for joint_index in range(frame_index + 1)]
            frames.append(self.__reduce_transformations(transformations, self.base_pose))

        return frames

    def fk_from_index_to_ee(self, index: int) -> SE2:
        transformations = [self.get_transformation_from_joint(joint_index) for joint_index in range(index, self.joint_count)]
        return self.__reduce_transformations(transformations, SE2(None, None))

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = []
        joint_transforms = [self.get_transformation_from_joint(i) for i in range(self.joint_count)]
        fk_all_links = self.fk_all_links()
        for index in range(len(joint_transforms)):
            if self.structure[index] == "P":
                jac_col = list(fk_all_links[index + 1].rotation.act((1, 0)))
                jac_col.append(0)
                jac.append(jac_col)
            else:
                T_je = self.__reduce_transformations(joint_transforms[index : ], SE2(None, None))
                t_je = T_je.translation
                n = SO2(np.pi / 2).act(t_je)
                n_s = list(fk_all_links[index].rotation.act(n))
                n_s.append(1)
                jac.append(n_s)

        return np.asarray(jac).transpose()
            
    
    def __finite_difference(self, delta: float, delta_vector, selector_function: Callable[[SE2], float]):
        copy_robot = deepcopy(self)
        copy_robot.q = np.add(copy_robot.q, delta_vector)
        return (selector_function(copy_robot.flange_pose()) - selector_function(self.flange_pose()))/ delta

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:

        to_delta_vector = lambda q_i: [delta if q == q_i else 0 for q in range(self.joint_count)]
        to_x: Callable[[SE2], float] = lambda t : t.translation[0]
        to_y: Callable[[SE2], float] = lambda t : t.translation[1]
        to_rot: Callable[[SE2], float] = lambda t : t.rotation.angle

        jac = []
        for i, selector in enumerate([to_x, to_y, to_rot]):
            jac.append([])
            for q_i in range(self.joint_count):
                jac[i].append(self.__finite_difference(delta, to_delta_vector(q_i), selector))

        return np.asarray(jac)

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW04 implement numerical IK

        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW04 implement analytical IK for RRR manipulator
        # todo: HW04 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            pass
        return []

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
    
    def __get_list_of_joint_types(self) -> list[str]:
        if type(self.structure) == str:
            return self.structure.split("")
        
        return self.structure
    
    def __reduce_transformations(self, transformations: list[SE2], base) -> SE2:
        return reduce(lambda se2_accumulator, se2 : se2_accumulator * se2, transformations, base)
    
    def get_transformation_from_joint(self, joint_index: int) -> SE2:
        q = self.q[joint_index]
        param = self.link_parameters[joint_index]
        joint_type = (self.__get_list_of_joint_types())[joint_index]
        if joint_type == "R":
            return self.get_se2_for_rotation_joint(q, param)
        else:
            return self.get_se2_for_prismatic_joint(q, param)

    @staticmethod
    def get_se2_for_prismatic_joint(extension: float, initial_rotation: float) -> SE2:
        translation = [extension, 0]
        return SE2(SO2(initial_rotation).act(translation), initial_rotation)
    
    @staticmethod
    def get_se2_for_rotation_joint(rotation: float, link_length: float) -> SE2:
        translation = [link_length, 0]
        return SE2(SO2(rotation).act(translation), rotation)         
