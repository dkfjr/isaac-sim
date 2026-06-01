# Cadev — LeKiwi Pick-and-Place (MimicGen 데이터 증강)

**LeKiwi**(메카넘 베이스 + SO-ARM100 팔) 로봇으로 빨간 큐브를 집어
검은 상자(`blackbox`) 위에 올려놓는 pick-and-place 태스크입니다.
**Isaac Sim 5.1 / IsaacLab + leisaac** 기반이며, 텔레옵으로 소수의 시연을
모은 뒤 **MimicGen** 으로 대량 증강합니다.

## 사전 요구사항

이 태스크는 [leisaac](https://github.com/LightwheelAI/leisaac) 프레임워크의
애드온입니다. LeKiwi 로봇, `tasks/template` 의 베이스 cfg, `enhance/envs`(Mimic env),
`leisaac.utils` 에 의존하므로, leisaac 체크아웃의
`source/leisaac/leisaac/tasks/` 안에 이 폴더를 넣어 사용하세요.

씬 USD 는 저장소 최상단 **`usd/`** 폴더에 있습니다(`usd/cadev_env.usd`, `usd/cadev.usd`).
태스크는 기본적으로 `<repo_root>/usd/cadev_env.usd` 를 씬 경로로 해석하며,
파일 위치가 다르면 `CADEV_USD_PATH` 환경변수로 덮어쓸 수 있습니다.

```bash
export CADEV_USD_PATH=/원하는/경로/cadev_env.usd
```

## USD 파일 (`cadev.usd` vs `cadev_env.usd`)

| 파일 | 구성 | 용도 |
|---|---|---|
| `usd/cadev.usd` | booth + blackbox + cube + **LeKiwi 로봇** + 카메라(Front, Wrist) | **작성용 전체 씬.** GUI 에서 배치를 디자인하고 로봇 초기 pos·카메라 pose 좌표를 추출하는 원본 |
| `usd/cadev_env.usd` | booth + blackbox + cube + Front 카메라 (**로봇 없음**) | **런타임에 태스크가 로드하는 씬**(`CADEV_USD_PATH`). 로봇은 IsaacLab 이 LeKiwi articulation 으로 따로 스폰하므로 중복 방지를 위해 env USD 에서는 로봇을 제외 |

## 태스크 ID

- `LeIsaac-LeKiwi-Cadev-v0` — 수집 / replay (`ManagerBasedRLEnv`, cfg: `CadevEnvCfg`)
- `LeIsaac-LeKiwi-Cadev-Mimic-v0` — annotate / generate (Mimic env, cfg: `CadevMimicEnvCfg`)

## 전체 워크플로

> 모든 명령은 leisaac 저장소 루트에서 실행한다고 가정합니다. 경로는 전부 상대경로이며,
> 사용자별 절대경로는 필요하지 않습니다.

### 1. 텔레옵으로 시연 수집 (9D joint action)
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-LeKiwi-Cadev-v0 --teleop_device lekiwi-leader \
  --enable_cameras --record \
  --dataset_file ./datasets/cadev_pnp.hdf5 --num_demos 8
```

### 2. IK 변환 (joint → EE pose, 8D)
```bash
python scripts/mimic/eef_action_process.py \
  --input_file ./datasets/cadev_pnp.hdf5 \
  --output_file ./datasets/cadev_pnp_ik.hdf5 --to_ik --headless
```

### 3. Annotate (subtask 경계 표시, 성공 시연만 통과)
```bash
python scripts/mimic/annotate_demos.py --device cuda \
  --task LeIsaac-LeKiwi-Cadev-Mimic-v0 \
  --input_file ./datasets/cadev_pnp_ik.hdf5 \
  --output_file ./datasets/cadev_pnp_annotated.hdf5 \
  --enable_cameras --auto
```

### 4. Generate (MimicGen 증강, seed 별로 다양성 확보)
```bash
for S in 42 7 100 777; do
  python scripts/mimic/generate_dataset.py --device cuda --num_envs 1 \
    --generation_num_trials 50 --seed $S \
    --task LeIsaac-LeKiwi-Cadev-Mimic-v0 \
    --input_file ./datasets/cadev_pnp_annotated.hdf5 \
    --output_file ./datasets/cadev_gen_s${S}.hdf5 --enable_cameras --headless
done
```

### 5. Merge (생성 파일 합치기)
```bash
python dependencies/IsaacLab/scripts/tools/merge_hdf5_datasets.py \
  --input_files ./datasets/cadev_gen_s42.hdf5 ./datasets/cadev_gen_s7.hdf5 \
                ./datasets/cadev_gen_s100.hdf5 ./datasets/cadev_gen_s777.hdf5 \
  --output_file ./datasets/cadev_gen_all.hdf5
```

### 6. to_joint 변환 (IK → joint, 학습 / 실로봇용)
```bash
python scripts/mimic/eef_action_process.py \
  --input_file ./datasets/cadev_gen_all.hdf5 \
  --output_file ./datasets/cadev_gen_all_joint.hdf5 --to_joint --headless
```

### 7. Replay 검증
```bash
python scripts/environments/teleoperation/replay.py \
  --task LeIsaac-LeKiwi-Cadev-v0 \
  --dataset_file ./datasets/cadev_gen_all_joint.hdf5 --enable_cameras
```

## 핵심 좌표 / 설정
- blackbox 윗면 z = 0.07, 중심 xy = (0.0, -0.005), 반폭 = (0.055, 0.038)
- 큐브 4cm (반높이 0.02). 박스 위 안착 시 큐브 중심 z ≈ 0.09
- 로봇 초기 pos = (0.0, 0.357, 0.039)
- 큐브 랜덤화: x(-0.03, 0.03) y(-0.03, 0.0) — 팔 도달 범위 내로 제한

## 성공 판정 (`mdp/terminations.py` — `cube_placed_on_box_and_rest`)
stateless 판정 (카운터 없음, trial 간 오염 방지):
1. 큐브 xy 가 박스 윗면적 범위 안 (어디든 OK)
2. 큐브 밑면(중심 − 0.02)이 박스 윗면(0.07)보다 위
3. 큐브 완전 정지 (전체 속도 < 0.02) — 떨어지거나 스쳐가는 중 배제

## 주의사항
- LeKiwi action 9D = `[arm5, gripper(idx5), wheel3]`. gripper 는 index 5 이고
  마지막 원소는 `base_theta`(SO101 과 다름). 항상 `joint_names.index("gripper")` 로 접근.
- Mimic IK action 8D = `[eef_pos3, eef_quat4, gripper1]`(휠 없음, 베이스는 stiffness 로 고정).
- mimic arm 게인: stiffness 500 / damping 90 (teleop 은 800 / 40).
- 공유 LeKiwi 로봇 USD 는 base/wheel collision 비활성화(booth floor 관통 방지),
  팔 collision 유지(파지용). ⚠️ 모든 LeKiwi 태스크가 공유하므로 수정 시 주의.
- `generate_dataset.py` 에 `--seed` 인자 추가됨(cfg 수정 없이 seed 변경).
- `generation_guarantee=True` 시 요청한 성공 개수 보장, `keep_failed=False` 로 실패본 미저장.
- generate 종료 시점에 나는 `'...MimicEnv' object has no attribute 'scene'` 에러는
  env 정리 단계의 것으로 생성된 데이터에는 영향 없음(`terminations.py` 에서 try/except 처리).
