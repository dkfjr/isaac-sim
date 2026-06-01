# Isaac Sim LeRobot Teleoperation Extensions

LeIsaac 프로젝트에 추가한 기능들입니다.

## 추가/수정 기능

### 1. 양팔 제어 (Bi-Arm Teleoperation)
- `bi_so101_leader.py`: 양팔 USB learder arm 지원
- `bi_so101_keyboard.py`: 양팔 키보드 teleop

### 2. 단팔 Joint 키보드 (`so101_joint_keyboard.py`)
- USB 없이 키보드로 joint position 방식 제어
- B키로 시작, ready pose로 자동 전환

### 3. USB ↔ 키보드 자동 전환 (`teleop_se3_agent.py`)
- USB 연결 해제 감지 → 키보드 전환 여부 묻기
- USB 재연결 감지 → USB 전환 여부 묻기
- N 선택 시 재감지 안 함 (USB 뽑혔다 꽂히면 다시 감지)
- 단팔(`so101leader`), 양팔(`bi-so101leader`) 모두 지원

### 4. Cadev — LeKiwi Pick-and-Place + MimicGen 데이터 증강
- LeKiwi(메카넘 베이스 + SO-ARM100 팔)로 빨간 큐브를 집어 검은 상자에 올리는 태스크
- 텔레옵 소수 시연 → IK 변환 → annotate → **MimicGen 으로 대량 증강** 파이프라인
- 추가 코드:
  - `source/leisaac/leisaac/tasks/cadev/` : 태스크 정의(env/mimic cfg, randomization, mdp)
  - `scripts/mimic/` : `eef_action_process.py`(IK↔joint 변환), `annotate_demos.py`, `generate_dataset.py`(`--seed` 인자 추가)
  - `source/leisaac/leisaac/devices/action_process.py` : `mimic_lekiwi-leader` 브랜치(팔 IK + 그리퍼, 휠 제외), wrist_roll 0점 보정
- 전체 워크플로/명령어/좌표·설정은 **[`tasks/cadev/README.md`](source/leisaac/leisaac/tasks/cadev/README.md)** 참고
- 씬 USD 는 `usd/cadev_env.usd`(런타임에 실제 쓰이는 씬). 코드는 `<repo_root>/usd/cadev_env.usd` 로 기본 해석하며 `CADEV_USD_PATH` 환경변수로 override 가능

## 키보드 조작법

### 단팔 (SO101JointKeyboard)
| 키 | 동작 |
|---|---|
| A/D | Shoulder Pan |
| W/S | Shoulder Lift |
| Q/E | Elbow |
| I/K | Wrist Flex |
| J/L | Wrist Roll |
| Z/X | Gripper |
| B | 시작 |

### 양팔 (BiSO101Keyboard)
| 키 | 동작 |
|---|---|
| A/D | Left Shoulder Pan |
| W/S | Left Shoulder Lift |
| Q/E | Left Elbow |
| I/K | Left Wrist Flex |
| J/L | Left Wrist Roll |
| Z/X | Left Gripper |
| LEFT/RIGHT | Right Shoulder Pan |
| UP/DOWN | Right Shoulder Lift |
| [/] | Right Elbow |
| ./SLASH | Right Wrist Flex |
| ;/' | Right Wrist Roll |
| M/, | Right Gripper |
| B | 시작 |

## 실행 방법

### 양팔 USB
```bash
./dependencies/IsaacLab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-SO101-LiftCube-BiArm-v0 \
  --teleop_device bi-so101leader \
  --left_arm_port /dev/ttyACM0 \
  --right_arm_port /dev/ttyACM1 \
  --num_envs 1 --enable_cameras
```

### 단팔 USB
```bash
./dependencies/IsaacLab/isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-SO101-LiftCube-v0 \
  --teleop_device so101leader \
  --port /dev/ttyACM0 \
  --num_envs 1 --enable_cameras
```

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `so101_leader.py` | 단팔 USB leader arm. 실제 robot arm의 joint position을 읽어 simulator에 전달 |
| `bi_so101_leader.py` | 양팔 USB leader arm. 좌/우 leader arm 동시 제어. Gripper calibration 포함 |
| `so101_keyboard.py` | 단팔 keyboard controller. 키보드로 robot arm의 방향/위치를 조작 |
| `so101_joint_keyboard.py` | 단팔 keyboard controller. Leader arm 없이 키보드로 각 joint를 직접 제어. so101leader fallback용 |
| `bi_so101_keyboard.py` | 양팔 keyboard controller. Leader arm 없이 키보드로 양팔 12 joints 제어. B키 후 ready pose 자동 전환 |
| `teleop_se3_agent.py` | Teleop 메인 스크립트. USB↔keyboard 자동 전환, episode 녹화, env reset 관리 |
| `action_process.py` | Teleop device의 입력을 env에 맞는 action 형식으로 변환 (단팔 6dim, 양팔 12dim) |

## USD 파일

`usd/` 폴더에 Isaac Sim용 USD 씬 파일과 생성 스크립트가 포함되어 있습니다.

### 씬 오브젝트

| 파일 | 설명 |
|---|---|
| `usd/booth.usd` | LeKiwi ㄷ자 부스. 바닥 60×55cm, 벽 높이 40cm, 두께 2cm. 정면(+Y) 트임, 천장 없음. Static collider (고정) |
| `usd/cube.usd` | 빨간 정육면체 (4×4×4cm, 50g). 파지 대상 오브젝트. Dynamic RigidBody로 중력 적용, 그리퍼로 집을 수 있음 |
| `usd/blackbox.usd` | 검은 상자 (11×7.5×5cm). 기본 Static (고정 구조물/받침대/장애물). 스크립트에서 Dynamic으로 전환 가능 |
| `usd/cadev_env.usd` | Cadev **런타임 씬** (booth + blackbox + cube, **로봇 없음**). `LeIsaac-LeKiwi-Cadev-v0` 가 로드하며 코드에서 `CADEV_USD_PATH` 로 참조. 텔레옵 수집·MimicGen 증강에 실제로 쓰이는 씬. 로봇은 IsaacLab 이 따로 스폰하므로 제외 |

### LeKiwi 로봇

| 파일 | 설명 |
|---|---|
| `usd/lekiwi/lekiwi_fixed.usd` | LeKiwi 로봇 메인 USD. base, physics, robot, sensor 레이어를 참조하는 합성 파일 |
| `usd/lekiwi/configuration/LeKiwi_base.usd` | 로봇 비주얼 메쉬 (외형 geometry) |
| `usd/lekiwi/configuration/LeKiwi_physics.usd` | 물리 시뮬레이션 설정 (joints, colliders) |
| `usd/lekiwi/configuration/LeKiwi_robot.usd` | 로봇 구조 정의 (articulation) |
| `usd/lekiwi/configuration/LeKiwi_sensor.usd` | 센서 구성 (카메라 등) |
| `usd/lekiwi/configuration/lekiwi_physics_*.usd` | physics 적용된 버전의 base/robot/sensor/physics 레이어 |

### USD 생성 스크립트

| 파일 | 설명 |
|---|---|
| `usd/make_booth_usd.py` | 부스 USD 생성. `python make_booth_usd.py --out ./booth.usd` |
| `usd/make_cube_usd.py` | 빨간 큐브 USD 생성. `python make_cube_usd.py --out ./cube.usd` |
| `usd/make_blackbox_usd.py` | 검은 상자 USD 생성. `python make_blackbox_usd.py --out ./blackbox.usd` |
