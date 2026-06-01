import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("RedCube"),
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """LeKiwi용 object_grasped (gripper 관절을 이름으로 인덱싱)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos = obj.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    gripper_idx = robot.data.joint_names.index("gripper")  # LeKiwi: 맨 끝이 아님
    grasped = torch.logical_and(
        pos_diff < diff_threshold,
        robot.data.joint_pos[:, gripper_idx] < grasp_threshold,
    )
    return grasped


def cube_on_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("RedCube"),
    box_center_xy: tuple = (0.0, -0.005),
    box_half_xy: tuple = (0.055, 0.038),
    box_top_z: float = 0.07,
    place_z_tol: float = 0.03,
    cube_speed_tol: float = 0.03,
) -> torch.Tensor:
    """큐브가 blackbox 위에 놓이고 정지했는지 (place 단계 종료 신호)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    origin = env.scene.env_origins
    cx = cube.data.root_pos_w[:, 0] - origin[:, 0]
    cy = cube.data.root_pos_w[:, 1] - origin[:, 1]
    cz = cube.data.root_pos_w[:, 2] - origin[:, 2]
    in_x = torch.abs(cx - box_center_xy[0]) < box_half_xy[0]
    in_y = torch.abs(cy - box_center_xy[1]) < box_half_xy[1]
    on_top = torch.abs(cz - (box_top_z + 0.02)) < place_z_tol
    settled = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=1) < cube_speed_tol
    return in_x & in_y & on_top & settled
