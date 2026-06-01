import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from scipy.spatial.transform import Rotation as _R
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from isaaclab.managers import EventTermCfg as EventTerm
from .randomization import reset_robot_facing_box, disable_booth_wall_collisions
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from . import mdp as cadev_mdp

from ..template import (
    LeKiwiObservationsCfg,
    LeKiwiTaskEnvCfg,
    LeKiwiTaskSceneCfg,
    LeKiwiTerminationsCfg,
)

# Path to the cadev scene USD. USD assets live in the repo's top-level ``usd/`` folder.
# Override with the ``CADEV_USD_PATH`` environment variable (e.g. when integrating this
# task into a leisaac checkout where the file lives elsewhere).
_DEFAULT_CADEV_USD = (Path(__file__).resolve().parents[5] / "usd" / "cadev_env.usd").as_posix()
CADEV_USD_PATH = os.environ.get("CADEV_USD_PATH", _DEFAULT_CADEV_USD)
CADEV_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=CADEV_USD_PATH)
)

def _euler_to_wxyz(rx, ry, rz):
    q = _R.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()  # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

FRONT_ROT = _euler_to_wxyz(82.296, -0.752, -0.102)
WRIST_ROT = _euler_to_wxyz(0.578, -1.606, 87.763)


@configclass
class CadevSceneCfg(LeKiwiTaskSceneCfg):
    """Scene configuration for the cadev task."""
    scene: AssetBaseCfg = CADEV_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.13, 0.025), rot=FRONT_ROT, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=27.0, horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0), lock_camera=True),
        width=640, height=480, update_period=1/30.0,
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.00356, 0.07455, -0.01838), rot=WRIST_ROT, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=27.0, horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0), lock_camera=True),
        width=640, height=480, update_period=1/30.0,
    )


@configclass
class CadevObservationsCfg(LeKiwiObservationsCfg):
    """LeKiwi 관측 + mimic subtask 신호 (잡기/놓기)."""

    @configclass
    class SubtaskCfg(ObsGroup):
        pick_cube = ObsTerm(
            func=cadev_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("RedCube"),
                "diff_threshold": 0.06,
                "grasp_threshold": 0.4,
            },
        )
        place_cube = ObsTerm(
            func=cadev_mdp.cube_on_box,
            params={"cube_cfg": SceneEntityCfg("RedCube")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class CadevTerminationsCfg(LeKiwiTerminationsCfg):
    """기본 종료 + 성공(놓기 + 그리퍼 놓음 + 휴식 복귀)."""

    success = DoneTerm(
        func=cadev_mdp.cube_placed_on_box_and_rest,
        params={
            "cube_cfg": SceneEntityCfg("RedCube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CadevEnvCfg(LeKiwiTaskEnvCfg):
    """Configuration for the cadev environment."""
    scene: CadevSceneCfg = CadevSceneCfg(env_spacing=8.0)
    observations: CadevObservationsCfg = CadevObservationsCfg()
    terminations: CadevTerminationsCfg = CadevTerminationsCfg()
    task_description: str = "Pick up the red cube in the booth."

    def __post_init__(self) -> None:
        super().__post_init__()
        # arm drive 게인 상향 (leader 추종 응답성 개선)
        self.scene.robot.actuators["sts3215-arm"].stiffness = 800.0
        self.scene.robot.actuators["sts3215-arm"].damping = 40.0
        # base 제자리 고정 (드리프트 방지)
        self.scene.robot.actuators["sts3215-base"].stiffness = 1000.0
        # 부스 앞 LeKiwi 위치 (cadev.usd 에서 추출한 값)
        self.scene.robot.init_state.pos = (0.0, 0.357, 0.039)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # 휴식 자세에서 시작 (use_default_offset=False 라 leader 추종에 영향 없음)
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan": -0.012,
            "shoulder_lift": -1.7376,
            "elbow_flex": 1.5192,
            "wrist_flex": 1.1193,
            "wrist_roll": 1.6465,
            "gripper": -0.1292,
        }

        # viewer 시점 (부스 쪽을 보게)
        self.viewer.eye = (-0.328, 0.484, 0.328)
        self.viewer.lookat = (0.0, -0.005, 0.04)

        # 큐브를 조작 가능한 entity로 등록
        parse_usd_and_create_subassets(CADEV_USD_PATH, self, specific_name_list=["cube"])

        # 르키위 좌우 이동 + 상자를 향해 회전 (front 카메라 다양한 각도 + 큐브 항상 프레임 내)
        self.events.randomize_robot_facing = EventTerm(
            func=reset_robot_facing_box,
            mode="reset",
            params={"x_range": (-0.1, 0.1), "box_xy": (0.0, -0.005)},
        )
        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "RedCube",
                    pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.0), "z": (0.0, 0.0)},
                ),
            ],
        )
