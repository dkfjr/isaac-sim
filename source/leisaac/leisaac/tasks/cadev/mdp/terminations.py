from __future__ import annotations
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_height_above_base(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """큐브가 로봇 base보다 height_threshold 이상 위로 올라가면 성공."""
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    return cube_height - robot_base_height > height_threshold


def cube_placed_on_box_and_rest(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("RedCube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_center_xy: tuple[float, float] = (0.0, -0.005),
    box_half_xy: tuple[float, float] = (0.055, 0.038),
    box_top_z: float = 0.07,
    place_z_tol: float = 0.02,
    cube_speed_tol: float = 0.02,
    gripper_open_thresh: float = 0.0,
    rest_joint_pos: dict | None = None,
    rest_tol: float = 0.30,
) -> torch.Tensor:
    """성공(stateless): 큐브가 blackbox 윗면적 위 + 밑면이 윗면보다 높음 + 완전 정지(안 떨어지고 안 스쳐감).
    카운터 없이 매 순간 독립 판정 → trial 간 상태 오염 없음."""
    # mimic env 는 self.scene 보유. 종료 시점엔 scene 이 해제될 수 있어 안전 처리.
    try:
        scene = env.scene
        cube: RigidObject = scene[cube_cfg.name]
        n_envs = scene.num_envs
    except Exception:
        # 종료 시점 등 scene 미접근 상황: 안전하게 False 1개 반환
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    cube_pos = cube.data.root_pos_w   # (N,3)
    cube_vel = cube.data.root_lin_vel_w  # (N,3)
    origin = scene.env_origins
    cx = cube_pos[:, 0] - origin[:, 0]
    cy = cube_pos[:, 1] - origin[:, 1]
    cz = cube_pos[:, 2] - origin[:, 2]

    cube_half = 0.02

    # 1) 큐브 xy 가 상자 윗면적 범위 안 (어디든 OK)
    in_x = torch.abs(cx - box_center_xy[0]) < box_half_xy[0]
    in_y = torch.abs(cy - box_center_xy[1]) < box_half_xy[1]
    # 2) 큐브 밑면이 상자 윗면보다 위 (상자 위에 얹힘)
    on_box = (cz - cube_half) > (box_top_z - 0.005)
    # 3) 완전 정지: 모든 방향 속도가 거의 0 (떨어지는/스쳐가는 중이면 속도 있음)
    speed = torch.linalg.vector_norm(cube_vel, dim=1)
    stopped = speed < cube_speed_tol

    return in_x & in_y & on_box & stopped
