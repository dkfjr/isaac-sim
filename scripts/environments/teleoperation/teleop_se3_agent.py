# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "bi-keyboard",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device in ["bi-so101leader", "bi-keyboard"], "only support bi-so101leader or bi-keyboard for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                if os.path.exists(args_cli.dataset_file):
                    answer = input(
                        "기존 파일이 존재합니다. 이어서 진행하시겠습니까? [Y/N]: "
                    ).strip()
                    if answer in ["y", "Y", "yes", "Yes"]:
                        args_cli.resume = True
                        env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                    else:
                        os.remove(args_cli.dataset_file)
                        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                else:
                    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager or lerobot recorder manager
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # create controller
    if args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader, SO101Keyboard
        teleop_interface = None
        for _retry in range(3):
            try:
                teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
                break
            except Exception as e:
                print(f"[WARN] SO101Leader 연결 실패 (시도 {_retry+1}/3): {e}")
                if _retry < 2:
                    import time as _t; _t.sleep(3.0)
        if teleop_interface is None:
            print("[INFO] 키보드로 전환합니다.")
            from leisaac.devices.keyboard.so101_joint_keyboard import SO101JointKeyboard
            teleop_interface = SO101JointKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "bi-keyboard":
        from leisaac.devices import BiSO101Keyboard
        teleop_interface = BiSO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader, BiSO101Keyboard
        teleop_interface = None
        for _retry in range(3):
            try:
                teleop_interface = BiSO101Leader(
                    env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate
                )
                break
            except Exception as e:
                print(f"[WARN] BiSO101Leader 연결 실패 (시도 {_retry+1}/3): {e}")
                if _retry < 2:
                    import time as _t; _t.sleep(2.0)
        if teleop_interface is None:
            print("[INFO] 키보드로 전환합니다.")
            teleop_interface = BiSO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    teleop_interface.display_controls()
    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        import h5py
        hdf5_path = args_cli.dataset_file
        with h5py.File(hdf5_path, 'a') as f:
            all_demos = sorted(f['data'].keys())
            failed = [d for d in all_demos if not f['data'][d].attrs.get('success', False)]
            for d in failed:
                del f['data'][d]
            remaining = sorted(f['data'].keys())
            for i, old_key in enumerate(remaining):
                if old_key != f'demo_{i}':
                    f['data'].move(old_key, f'demo_{i}')
            resume_recorded_demo_count = len(list(f['data'].keys()))
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count
    dataset_file_path = args_cli.dataset_file

    start_record_state = False

    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    import threading
    from leisaac.devices import SO101Keyboard as _SO101KB, BiSO101Keyboard as _BiSO101KB
    from leisaac.devices.keyboard.so101_joint_keyboard import SO101JointKeyboard as _SO101JKB
    _using_keyboard_fallback = (
        (args_cli.teleop_device == "bi-so101leader" and isinstance(teleop_interface, _BiSO101KB)) or
        (args_cli.teleop_device == "so101leader" and isinstance(teleop_interface, (_SO101KB, _SO101JKB)))
    )
    _usb_retry_counter = 0
    _usb_retry_interval = 300
    _usb_asking = False
    _usb_answer = None
    _usb_declined = False  # N 누른 후 재감지 방지

    def _ask_user(msg):
        nonlocal _usb_answer
        ans = input(msg).strip()
        _usb_answer = ans

    try:
        while simulation_app.is_running() and not interrupted:
            # run everything in inference mode
            with torch.inference_mode():
                # USB 연결 상태 관리 (so101leader)
                if args_cli.teleop_device == "so101leader" and not _usb_asking:
                    _usb_retry_counter += 1
                    if _usb_retry_counter >= _usb_retry_interval:
                        _usb_retry_counter = 0
                        usb_present = os.path.exists(args_cli.port)
                        if _usb_declined and not usb_present:
                            _usb_declined = False
                        if _using_keyboard_fallback and usb_present and not _usb_declined:
                            _usb_asking = True
                            _usb_answer = None
                            threading.Thread(target=_ask_user, args=("USB가 감지되었습니다. USB로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()
                        elif not _using_keyboard_fallback and not usb_present:
                            _usb_asking = True
                            _usb_answer = None
                            threading.Thread(target=_ask_user, args=("USB 연결이 끊겼습니다. 키보드로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()

                # USB 연결 상태 관리 (bi-so101leader)
                if args_cli.teleop_device == "bi-so101leader" and not _usb_asking:
                    _usb_retry_counter += 1
                    if _usb_retry_counter >= _usb_retry_interval:
                        _usb_retry_counter = 0
                        usb_present = os.path.exists(args_cli.left_arm_port) and os.path.exists(args_cli.right_arm_port)
                        if _usb_declined and not usb_present:
                            _usb_declined = False
                        if _using_keyboard_fallback and usb_present and not _usb_declined:
                            _usb_asking = True
                            _usb_answer = None
                            threading.Thread(target=_ask_user, args=("USB가 감지되었습니다. USB로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()
                        elif not _using_keyboard_fallback and not usb_present:
                            _usb_asking = True
                            _usb_answer = None
                            threading.Thread(target=_ask_user, args=("USB 연결이 끊겼습니다. 키보드로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()

                if _usb_asking and _usb_answer is not None and args_cli.teleop_device == "so101leader":
                    if _using_keyboard_fallback:
                        if _usb_answer in ["Y", "y", "Yes", "yes"]:
                            try:
                                try:
                                    teleop_interface._stop_keyboard_listener()
                                except Exception:
                                    pass
                                __import__("time").sleep(5.0)
                                from leisaac.devices import SO101Leader
                                new_if = SO101Leader(env, port=args_cli.port, recalibrate=False)
                                new_if.add_callback("R", reset_recording_instance)
                                new_if.add_callback("N", reset_task_success)
                                teleop_interface = new_if
                                teleop_interface.reset()
                                env.reset()
                                _using_keyboard_fallback = False
                                teleop_interface._started = False
                                _usb_retry_counter = 0
                                print("[INFO] SO101Leader로 전환되었습니다. B를 눌러 시작하세요.")
                            except Exception as e:
                                print(f"[WARN] USB 연결 실패: {e}")
                        else:
                            print("[INFO] 키보드로 계속합니다.")
                            _usb_declined = True
                    else:
                        if _usb_answer in ["Y", "y", "Yes", "yes"]:
                            try:
                                teleop_interface.disconnect()
                            except Exception:
                                pass
                            from leisaac.devices.keyboard.so101_joint_keyboard import SO101JointKeyboard
                            teleop_interface = SO101JointKeyboard(env, sensitivity=args_cli.sensitivity)
                            teleop_interface.add_callback("R", reset_recording_instance)
                            teleop_interface.add_callback("N", reset_task_success)
                            _using_keyboard_fallback = True
                            teleop_interface._started = False
                            _usb_retry_counter = 0
                            env.reset()
                            print("[INFO] 키보드로 전환되었습니다. B를 눌러 시작하세요.")
                        else:
                            print("[INFO] 종료합니다.")
                            interrupted = True
                    _usb_asking = False
                    _usb_answer = None
                    _usb_retry_counter = 0

                if _usb_asking and _usb_answer is not None and args_cli.teleop_device in ["bi-so101leader"]:
                    if _using_keyboard_fallback:
                        if _usb_answer in ["Y", "y", "Yes", "yes"]:
                            try:
                                # 기존 키보드 인터페이스 정리
                                try:
                                    teleop_interface._stop_keyboard_listener()
                                except Exception:
                                    pass
                                __import__("time").sleep(2.0)
                                from leisaac.devices import BiSO101Leader
                                new_interface = BiSO101Leader(
                                    env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=False
                                )
                                new_interface.add_callback("R", reset_recording_instance)
                                new_interface.add_callback("N", reset_task_success)
                                teleop_interface = new_interface
                                teleop_interface.reset()
                                env.reset()
                                _using_keyboard_fallback = False
                                teleop_interface._started = False
                                _usb_retry_counter = 0
                                print("[INFO] Leader arm으로 전환되었습니다. B를 눌러 시작하세요.")
                            except Exception as e:
                                print(f"[WARN] USB 연결 실패: {e}")
                                _usb_asking = False
                                _usb_answer = None
                                _usb_retry_counter = 0
                        else:
                            print("[INFO] 키보드로 계속합니다.")
                            _usb_declined = True
                    else:
                        if _usb_answer in ["Y", "y", "Yes", "yes"]:
                            # 기존 USB 포트 닫기
                            try:
                                teleop_interface.left_so101_leader.disconnect()
                                teleop_interface.right_so101_leader.disconnect()
                            except Exception:
                                pass
                            from leisaac.devices import BiSO101Keyboard
                            teleop_interface = BiSO101Keyboard(env, sensitivity=args_cli.sensitivity)
                            teleop_interface.add_callback("R", reset_recording_instance)
                            teleop_interface.add_callback("N", reset_task_success)
                            _using_keyboard_fallback = True
                            teleop_interface._started = False
                            _usb_retry_counter = 0
                            env.reset()
                            print("[INFO] 키보드로 전환되었습니다. B를 눌러 시작하세요.")
                        else:
                            print("[INFO] 종료합니다.")
                            interrupted = True
                    _usb_asking = False
                    _usb_answer = None
                    _usb_retry_counter = 0



                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                try:
                    actions = teleop_interface.advance()
                except Exception as e:
                    if not _using_keyboard_fallback and args_cli.teleop_device == "so101leader" and not _usb_asking:
                        _usb_asking = True
                        _usb_answer = None
                        _usb_retry_counter = 0
                        print(f"[WARN] USB 연결 끊김: {e}")
                        threading.Thread(target=_ask_user, args=("키보드로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()
                    elif not _using_keyboard_fallback and args_cli.teleop_device == "bi-so101leader" and not _usb_asking:
                        _usb_asking = True
                        _usb_answer = None
                        _usb_retry_counter = 0
                        print(f"[WARN] USB 연결 끊김: {e}")
                        threading.Thread(target=_ask_user, args=("키보드로 전환하시겠습니까? [Y/N]: ",), daemon=True).start()
                    actions = None
                if should_reset_task_success:
                    should_reset_task_success = False
                    if args_cli.record:
                        manual_terminate(env, True)
                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    if start_record_state:
                        start_record_state = False
                    if args_cli.record:
                        manual_terminate(env, False)
                    # print out the current demo count if it has changed
                    if (
                        args_cli.record
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        > current_recorded_demo_count
                    ):
                        current_recorded_demo_count = (
                            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        )
                        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                    if (
                        args_cli.record
                        and args_cli.num_demos > 0
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        >= args_cli.num_demos
                    ):
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                        break

                elif actions is None:
                    env.render()
                # apply actions
                else:
                    if not start_record_state:
                        if args_cli.record:
                            ep_num = env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                            print(f"Recording Episode {ep_num}")
                        start_record_state = True
                    env.step(actions)
                if rate_limiter:
                    rate_limiter.sleep(env)
            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # finalize the recorder manager
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        if args_cli.record and os.path.exists(args_cli.dataset_file):
            import shutil, h5py
            with h5py.File(args_cli.dataset_file, 'a') as f:
                demos = sorted(f['data'].keys())
                for i, old_key in enumerate(demos):
                    if old_key != f'demo_{i}':
                        f['data'].move(old_key, f'demo_{i}')
            host_dir = "/workspace/leisaac/datasets"
            dest = os.path.join(host_dir, os.path.basename(args_cli.dataset_file))
            if os.path.abspath(args_cli.dataset_file) != os.path.abspath(dest):
                shutil.copy2(args_cli.dataset_file, dest)
            os.chmod(dest, 0o666)
            print(f"[INFO] 데이터셋이 호스트에 저장되었습니다: {dest}")
        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
