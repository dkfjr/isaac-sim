import carb
import numpy as np
from ..device_base import Device

class SO101JointKeyboard(Device):
    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "so101_joint_keyboard")
        self.joint_sensitivity = 0.08 * sensitivity
        self._joint_offsets = np.zeros(6)
        self._joint_offsets[5] = -0.174
        self._READY_POSE = np.array([0.0, -1.745, 1.560, 0.873, 0.0, 0.0])
        self._startup_counter = 0

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["A/D", "shoulder pan"])
        self._display_controls_table.add_row(["W/S", "shoulder lift"])
        self._display_controls_table.add_row(["Q/E", "elbow"])
        self._display_controls_table.add_row(["I/K", "wrist flex"])
        self._display_controls_table.add_row(["J/L", "wrist roll"])
        self._display_controls_table.add_row(["Z/X", "gripper open/close"])

    def get_device_state(self):
        b_key = carb.input.KeyboardInput.B
        if self._input.get_keyboard_value(self._keyboard, b_key) > 0 and not self._started:
            self._started = True
            self._startup_counter = 20

        if not self._started:
            return np.zeros(6)

        if self._startup_counter > 0:
            t = 1.0 - (self._startup_counter / 20.0)
            self._joint_offsets[:] = self._READY_POSE * t
            self._joint_offsets[5] = -0.174
            self._startup_counter -= 1
            return self._joint_offsets.copy()

        s = self.joint_sensitivity
        kv = self._input.get_keyboard_value
        kb = self._keyboard

        def k(key_name):
            return kv(kb, getattr(carb.input.KeyboardInput, key_name)) > 0

        if k("A"): self._joint_offsets[0] -= s
        if k("D"): self._joint_offsets[0] += s
        if k("W"): self._joint_offsets[1] += s
        if k("S"): self._joint_offsets[1] -= s
        if k("Q"): self._joint_offsets[2] -= s
        if k("E"): self._joint_offsets[2] += s
        if k("I"): self._joint_offsets[3] -= s
        if k("K"): self._joint_offsets[3] += s
        if k("J"): self._joint_offsets[4] -= s
        if k("L"): self._joint_offsets[4] += s
        if k("Z"): self._joint_offsets[5] += s
        if k("X"): self._joint_offsets[5] -= s

        return self._joint_offsets.copy()

    def reset(self):
        self._joint_offsets = np.zeros(6)
        self._joint_offsets[5] = -0.174
        self._startup_counter = 0
        self._started = False
