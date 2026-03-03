import carb
import numpy as np

from ..device_base import Device


class BiSO101Keyboard(Device):
    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "bi_keyboard")
        self.pos_sensitivity = 0.05 * sensitivity
        self.joint_sensitivity = 0.08 * sensitivity
        self.rot_sensitivity = 0.5 * sensitivity
        self._create_key_bindings()
        self._startup_counter = 0
        # 누적 offset (RelativeJointPos는 매 프레임 default에 더하므로 누적값 유지)
        self._joint_offsets = __import__("numpy").zeros(12)
        self._joint_offsets[5] = -0.174   # left gripper closed
        self._joint_offsets[11] = -0.174  # right gripper closed
        # ready pose offsets
        self._READY_POSE = __import__("numpy").array([
            0.0, -1.745, 1.560, 0.873, 0.0, 0.0,  # left (gripper=0: init_state에서 이미 설정)
            0.0, -1.745, 1.560, 0.873, 0.0, 0.0,  # right
        ])
        self._delta_action = __import__("numpy").zeros(12)
        self.left_robot = self.env.scene["left_arm"]
        self.right_robot = self.env.scene["right_arm"]
        self.target_frame = "gripper"
        l, _ = self.left_robot.find_bodies(self.target_frame)
        r, _ = self.right_robot.find_bodies(self.target_frame)
        self.left_target_frame_idx = l[0]
        self.right_target_frame_idx = r[0]

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["W/S", "Left forward/backward"])
        self._display_controls_table.add_row(["A/D", "Left left/right"])
        self._display_controls_table.add_row(["Q/E", "Left up/down"])
        self._display_controls_table.add_row(["J/L", "Left rotate left/right"])
        self._display_controls_table.add_row(["I/K", "Left rotate up/down"])
        self._display_controls_table.add_row(["Z/X", "Left gripper open/close"])
        self._display_controls_table.add_row(["UP/DOWN", "Right forward/backward"])
        self._display_controls_table.add_row(["LEFT/RIGHT", "Right left/right"])
        self._display_controls_table.add_row(["[/]", "Right up/down"])
        self._display_controls_table.add_row(["SEMICOLON/APOSTROPHE", "Right rotate left/right"])
        self._display_controls_table.add_row(["PERIOD/SLASH", "Right rotate up/down"])
        self._display_controls_table.add_row(["M/,", "Right gripper open/close"])
    def get_device_state(self):
        b_key = carb.input.KeyboardInput.B
        if self._input.get_keyboard_value(self._keyboard, b_key) > 0 and not self._started:
            self._started = True
            self._startup_counter = 20
            self._joint_offsets[:] = 0.0
            self._joint_offsets[5] = -0.174
            self._joint_offsets[11] = -0.174

        if self._startup_counter > 0:
            self._startup_counter -= 1
            # 60프레임에 걸쳐 ready pose로 이동
            step = self._READY_POSE / 20.0
            self._joint_offsets += step

        elif self._started:
            per_frame_delta = __import__("numpy").zeros(12)
            for key_name, action in self._INPUT_KEY_MAPPING.items():
                key = getattr(carb.input.KeyboardInput, key_name)
                if self._input.get_keyboard_value(self._keyboard, key) > 0:
                    per_frame_delta += self._ACTION_DELTA_MAPPING[action]
            self._joint_offsets += per_frame_delta

        return {"left_arm": self._joint_offsets[:6], "right_arm": self._joint_offsets[6:]}

    def reset(self):
        self._delta_action[:] = 0.0

    def _create_key_bindings(self):
        j = self.joint_sensitivity
        z = __import__("numpy").zeros(12)
        def a(idx, val):
            v = z.copy(); v[idx] = val; return v
        self._ACTION_DELTA_MAPPING = {
            "l_pan_left":  a(0,-1.0)*j, "l_pan_right":  a(0, 1.0)*j,
            "l_lift_up":   a(1, 1.0)*j, "l_lift_down":  a(1,-1.0)*j,
            "l_elbow_up":  a(2, 1.0)*j, "l_elbow_down": a(2,-1.0)*j,
            "l_wrist_up":  a(3, 1.0)*j, "l_wrist_down": a(3,-1.0)*j,
            "l_roll_left": a(4,-1.0)*j, "l_roll_right": a(4, 1.0)*j,
            "l_grip_open": a(5, 1.0)*j, "l_grip_close": a(5,-1.0)*j,
            "r_pan_left":  a(6,-1.0)*j, "r_pan_right":  a(6, 1.0)*j,
            "r_lift_up":   a(7, 1.0)*j, "r_lift_down":  a(7,-1.0)*j,
            "r_elbow_up":  a(8, 1.0)*j, "r_elbow_down": a(8,-1.0)*j,
            "r_wrist_up":  a(9, 1.0)*j, "r_wrist_down": a(9,-1.0)*j,
            "r_roll_left": a(10,-1.0)*j,"r_roll_right": a(10, 1.0)*j,
            "r_grip_open": a(11, 1.0)*j,"r_grip_close": a(11,-1.0)*j,
        }
        self._INPUT_KEY_MAPPING = {
            "A":"l_pan_left",   "D":"l_pan_right",
            "W":"l_lift_up",    "S":"l_lift_down",
            "Q":"l_elbow_up",   "E":"l_elbow_down",
            "I":"l_wrist_up",   "K":"l_wrist_down",
            "J":"l_roll_left",  "L":"l_roll_right",
            "Z":"l_grip_open",  "X":"l_grip_close",
            "LEFT":"r_pan_left",  "RIGHT":"r_pan_right",
            "UP":"r_lift_up",     "DOWN":"r_lift_down",
            "LEFT_BRACKET":"r_elbow_up",  "RIGHT_BRACKET":"r_elbow_down",
            "PERIOD":"r_wrist_up",         "SLASH":"r_wrist_down",
            "SEMICOLON":"r_roll_left",     "APOSTROPHE":"r_roll_right",
            "M":"r_grip_open",  "COMMA":"r_grip_close",
        }
