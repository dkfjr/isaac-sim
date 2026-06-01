from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .cadev_env_cfg import CadevEnvCfg


@configclass
class CadevMimicEnvCfg(CadevEnvCfg, MimicEnvCfg):
    """cadev (LeKiwi pick-and-place) mimic 환경 설정."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "cadev_lekiwi_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        subtask_configs = []

        # 1) 큐브 잡기 (pick_cube 신호가 끝 경계)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="RedCube",
                subtask_term_signal="pick_cube",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pick cube",
                next_subtask_description="Place cube on box",
            )
        )

        # 2) blackbox 위에 놓기 (place_cube 신호가 끝 경계)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="RedCube",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube on box",
                next_subtask_description="Return to rest",
            )
        )

        self.subtask_configs["lekiwi_follower"] = subtask_configs

        # 떨림 완화: IK mimic 에서는 텔레옵용 stiffness(800) 가 과해서 진동 발생.
        # stiffness 낮추고 damping 올려 IK 목표 추종 시 과민반응/진동 억제.
        self.scene.robot.actuators["sts3215-arm"].stiffness = 500.0
        self.scene.robot.actuators["sts3215-arm"].damping = 90.0
