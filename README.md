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
