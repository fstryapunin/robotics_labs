from pathlib import Path
from robotics_toolbox.robots import SpatialManipulator
from robotics_toolbox.render import RendererSpatial

renderer = RendererSpatial()
robot = SpatialManipulator(urdf_path=Path(__file__).parent.joinpath("robot_hw.urdf"))

renderer.plot_manipulator(robot)
frame = robot.flange_pose()
renderer.plot_se3(frame)
