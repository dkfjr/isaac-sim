import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


def reset_robot_facing_box(env, env_ids, x_range, box_xy, asset_cfg=SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    num = len(env_ids)
    default_root = asset.data.default_root_state[env_ids].clone()

    # 좌우(X) 랜덤 이동
    x_off = torch.empty(num, device=device).uniform_(float(x_range[0]), float(x_range[1]))
    pos = default_root[:, :3].clone()
    pos[:, 0] += x_off
    pos += env.scene.env_origins[env_ids]

    # 상자를 향해 회전 (좌우로 간 만큼 yaw 보정)
    rel_y = default_root[:, 1] - box_xy[1]
    add_yaw = -torch.atan2(x_off, rel_y)   # 방향이 반대면 -torch.atan2(...) 로
    zeros = torch.zeros(num, device=device)
    dyaw_quat = quat_from_euler_xyz(zeros, zeros, add_yaw)
    quat = quat_mul(default_root[:, 3:7], dyaw_quat)

    root_pose = torch.cat([pos, quat], dim=-1)
    asset.write_root_pose_to_sim(root_pose, env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num, 6, device=device), env_ids)


def disable_booth_wall_collisions(env, env_ids=None):
    """robot 과 booth floor 사이의 충돌만 무시 (collision filtering).
    floor 는 큐브/박스를 계속 받치고, robot 바퀴만 floor 를 통과한다.
    PhysX filtered pairs: floor prim 에 robot prim 들을 filtered pair 로 추가."""
    from pxr import Usd, UsdPhysics, Sdf, PhysxSchema
    stage = env.sim.stage

    disabled = 0
    for env_idx in range(env.num_envs):
        floor_path = f"/World/envs/env_{env_idx}/Scene/booth/Booth/floor"
        robot_path = f"/World/envs/env_{env_idx}/Robot"
        floor_prim = stage.GetPrimAtPath(floor_path)
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not (floor_prim and floor_prim.IsValid()):
            continue
        if not (robot_prim and robot_prim.IsValid()):
            continue
        # floor 에 PhysxCollisionAPI 의 filteredPairs 로 robot 추가
        try:
            filt_api = PhysxSchema.PhysxCollisionAPI.Apply(floor_prim)
        except Exception:
            pass
        # filteredPairs 관계(rel) 사용: UsdPhysics 의 filteredPairs
        rel = floor_prim.GetRelationship("physics:filteredPairs")
        if not rel:
            rel = floor_prim.CreateRelationship("physics:filteredPairs", custom=False)
        targets = list(rel.GetTargets())
        if Sdf.Path(robot_path) not in targets:
            targets.append(Sdf.Path(robot_path))
            rel.SetTargets(targets)
            disabled += 1
    print(f"===== [BOOTH FILTER] {disabled}개 env 에서 robot<->floor 충돌 필터링 적용 =====")
