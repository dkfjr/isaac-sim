# Cadev — LeKiwi Pick-and-Place (MimicGen data augmentation)

A pick-and-place task where the **LeKiwi** robot (mecanum base + SO-ARM100 arm)
grasps a red cube and places it on top of a black box (`blackbox`).
Built on **Isaac Sim 5.1 / IsaacLab + leisaac**. A handful of teleoperated
demonstrations are collected, then amplified into a large dataset with **MimicGen**.

## Requirements

This task is an add-on to the [leisaac](https://github.com/LightwheelAI/leisaac)
framework and relies on it for the LeKiwi robot, the `tasks/template` base
configs, `enhance/envs` (the Mimic env), and `leisaac.utils`. To use it, drop
this folder into `source/leisaac/leisaac/tasks/` of a leisaac checkout.

The scene USD lives in the repo's top-level **`usd/`** folder
(`usd/cadev_env.usd`, `usd/cadev.usd`). The task resolves the scene path from
`<repo_root>/usd/cadev_env.usd` by default; override it with the
`CADEV_USD_PATH` environment variable if your file is elsewhere:

```bash
export CADEV_USD_PATH=/path/to/your/cadev_env.usd
```

## Task IDs

- `LeIsaac-LeKiwi-Cadev-v0` — collection / replay (`ManagerBasedRLEnv`, cfg: `CadevEnvCfg`)
- `LeIsaac-LeKiwi-Cadev-Mimic-v0` — annotate / generate (Mimic env, cfg: `CadevMimicEnvCfg`)

## Full workflow

> All commands assume you run from the leisaac repo root. Paths are relative; no
> absolute/user-specific paths are required.

### 1. Collect demonstrations by teleoperation (9D joint action)
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-LeKiwi-Cadev-v0 --teleop_device lekiwi-leader \
  --enable_cameras --record \
  --dataset_file ./datasets/cadev_pnp.hdf5 --num_demos 8
```

### 2. IK conversion (joint → EE pose, 8D)
```bash
python scripts/mimic/eef_action_process.py \
  --input_file ./datasets/cadev_pnp.hdf5 \
  --output_file ./datasets/cadev_pnp_ik.hdf5 --to_ik --headless
```

### 3. Annotate (mark subtask boundaries; only successful demos pass)
```bash
python scripts/mimic/annotate_demos.py --device cuda \
  --task LeIsaac-LeKiwi-Cadev-Mimic-v0 \
  --input_file ./datasets/cadev_pnp_ik.hdf5 \
  --output_file ./datasets/cadev_pnp_annotated.hdf5 \
  --enable_cameras --auto
```

### 4. Generate (MimicGen augmentation; vary the seed for diversity)
```bash
for S in 42 7 100 777; do
  python scripts/mimic/generate_dataset.py --device cuda --num_envs 1 \
    --generation_num_trials 50 --seed $S \
    --task LeIsaac-LeKiwi-Cadev-Mimic-v0 \
    --input_file ./datasets/cadev_pnp_annotated.hdf5 \
    --output_file ./datasets/cadev_gen_s${S}.hdf5 --enable_cameras --headless
done
```

### 5. Merge the generated files
```bash
python dependencies/IsaacLab/scripts/tools/merge_hdf5_datasets.py \
  --input_files ./datasets/cadev_gen_s42.hdf5 ./datasets/cadev_gen_s7.hdf5 \
                ./datasets/cadev_gen_s100.hdf5 ./datasets/cadev_gen_s777.hdf5 \
  --output_file ./datasets/cadev_gen_all.hdf5
```

### 6. Convert back to joint (IK → joint, for training / real robot)
```bash
python scripts/mimic/eef_action_process.py \
  --input_file ./datasets/cadev_gen_all.hdf5 \
  --output_file ./datasets/cadev_gen_all_joint.hdf5 --to_joint --headless
```

### 7. Replay to verify
```bash
python scripts/environments/teleoperation/replay.py \
  --task LeIsaac-LeKiwi-Cadev-v0 \
  --dataset_file ./datasets/cadev_gen_all_joint.hdf5 --enable_cameras
```

## Key coordinates / settings
- blackbox top z = 0.07, center xy = (0.0, -0.005), half-extent = (0.055, 0.038)
- cube 4 cm (half-height 0.02). When resting on the box, cube center z ≈ 0.09
- robot initial pos = (0.0, 0.357, 0.039)
- cube randomization: x(-0.03, 0.03) y(-0.03, 0.0) — kept within the arm's reach

## Success criterion (`mdp/terminations.py` — `cube_placed_on_box_and_rest`)
Stateless check (no counters, so no cross-trial contamination):
1. cube xy is within the box's top face (anywhere is OK)
2. cube bottom (center − 0.02) is above the box top (0.07)
3. cube is fully at rest (total speed < 0.02) — excludes falling / grazing-through

## Notes
- LeKiwi action 9D = `[arm5, gripper(idx5), wheel3]`. The gripper is at index 5 and
  the last element is `base_theta` (different from SO101). Always access it via
  `joint_names.index("gripper")`.
- Mimic IK action 8D = `[eef_pos3, eef_quat4, gripper1]` (no wheels; the base is held
  by joint stiffness).
- mimic arm gains: stiffness 500 / damping 90 (teleop uses 800 / 40).
- The shared LeKiwi robot USD has base/wheel collision disabled (to avoid clipping
  through the booth floor) while the arm collision stays on (for grasping).
  ⚠️ It is shared across all LeKiwi tasks — change it with care.
- `generate_dataset.py` adds a `--seed` argument (vary the seed without editing the cfg).
- With `generation_guarantee=True` the requested number of successes is guaranteed;
  `keep_failed=False` skips saving failed trials.
- The `'...MimicEnv' object has no attribute 'scene'` error at the end of generation
  comes from the env-teardown stage and does not affect the generated data
  (handled with try/except in `terminations.py`).
