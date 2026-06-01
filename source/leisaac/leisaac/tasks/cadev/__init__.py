import gymnasium as gym

gym.register(
    id="LeIsaac-LeKiwi-Cadev-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cadev_env_cfg:CadevEnvCfg",
    },
)

gym.register(
    id="LeIsaac-LeKiwi-Cadev-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cadev_mimic_env_cfg:CadevMimicEnvCfg",
    },
)
