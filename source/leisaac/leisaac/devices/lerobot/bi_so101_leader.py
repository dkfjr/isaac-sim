from ..device_base import Device
from .so101_leader import SO101Leader


class BiSO101Leader(Device):
    def __init__(
        self, env, left_port: str = "/dev/ttyACM0", right_port: str = "/dev/ttyACM1", recalibrate: bool = False
    ):
        super().__init__(env, "bi_so101_leader")

        # use left so101 leader as the main device to store state
        print("Connecting to left_so101_leader...")
        self.left_so101_leader = SO101Leader(env, left_port, recalibrate, "left_so101_leader.json")
        print("Connecting to right_so101_leader...")
        self.right_so101_leader = SO101Leader(env, right_port, recalibrate, "right_so101_leader.json")

        self.left_so101_leader._stop_keyboard_listener()
        self.right_so101_leader._stop_keyboard_listener()

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["bi-so101-leader", "move bi-so101-leader to control bi-so101-follower"])
        self._display_controls_table.add_row([
            "[TIPS]",
            (
                "If Bi-SO101-Follower can't synchronize with Bi-SO101-Leader, please add --recalibrate and rerun to"
                " recalibrate Bi-SO101-Leader"
            ),
        ])

    def reset(self):
        self.left_so101_leader.reset()
        self.right_so101_leader.reset()
        super().reset()

    def get_device_state(self):
        import carb
        if not self._started:
            b_key = carb.input.KeyboardInput.B
            if self._input.get_keyboard_value(self._keyboard, b_key) > 0:
                self._started = True
        left_state = self.left_so101_leader.get_device_state()
        right_state = self.right_so101_leader.get_device_state()
        # 왼쪽 그리퍼 오프셋: 첫 프레임에 오른쪽과의 차이를 기록
        if not hasattr(self, "_gripper_offset_calibrated"):
            self._gripper_offset_calibrated = True
            self._left_gripper_offset = right_state["gripper"] - left_state["gripper"]
        left_state["gripper"] = left_state["gripper"] + self._left_gripper_offset
        return {
            "left_arm": left_state,
            "right_arm": right_state,
        }

    def input2action(self):
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = {
            "left_arm": self.left_so101_leader.motor_limits,
            "right_arm": self.right_so101_leader.motor_limits,
        }
        return ac_dict
