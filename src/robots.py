"""xArm picker and sorter controllers."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config import (
    HANDOFF_PITCH,
    HANDOFF_ROLL,
    HANDOFF_X,
    HANDOFF_Y,
    HANDOFF_YAW,
    HANDOFF_Z,
    R2_APPROVED_PITCH,
    R2_APPROVED_ROLL,
    R2_APPROVED_X,
    R2_APPROVED_Y,
    R2_APPROVED_YAW,
    R2_APPROVED_Z,
    R2_HANDOFF_PITCH,
    R2_HANDOFF_ROLL,
    R2_HANDOFF_X,
    R2_HANDOFF_Y,
    R2_HANDOFF_YAW,
    R2_HANDOFF_Z,
    R2_REJECTED_PITCH,
    R2_REJECTED_ROLL,
    R2_REJECTED_X,
    R2_REJECTED_Y,
    R2_REJECTED_YAW,
    R2_REJECTED_Z,
    R2_X_RETRACT,
    R2_Y_RETRACT,
    R2_Z_SAFE,
    X_RETRACT,
    Y_RETRACT,
    Z_GRASP_MIN,
    Z_HOME_SEGURO,
    Z_OFFSET_AGARRE,
)

log = logging.getLogger(__name__)

try:
    from xarm.wrapper import XArmAPI

    ROBOT_DISPONIBLE = True
except ImportError:
    XArmAPI = None
    ROBOT_DISPONIBLE = False
    log.warning("xarm-python-sdk not installed — robots will be simulated.")


class RobotMain:
    """Controls Robot 1: table pickup and delivery to the handoff point."""

    def __init__(self, robot_ip: str, matrix_file: str):
        self._ip = robot_ip
        self._matrix_file = matrix_file
        self._arm = None
        self._tcp_speed = 100
        self.matriz_vision: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    @property
    def ip(self) -> str:
        return self._ip

    def connect(self) -> bool:
        if not ROBOT_DISPONIBLE:
            log.info("Robot SDK unavailable — Robot 1 is in simulation mode.")
            return False
        try:
            self._arm = XArmAPI(self._ip, baud_checkset=False)
            self._robot_init()
            self._load_calibration_matrix(self._matrix_file)
            log.info("Robot 1 connected and ready at %s.", self._ip)
            return True
        except Exception as exc:
            log.error("Robot 1 connection failed: %s", exc)
            self._arm = None
            return False

    def pick_and_place(
        self, x_pix: int, y_pix: int, z_vision: float
    ) -> tuple[float | None, float | None]:
        """Pick from the table, place at handoff and return robot XY."""
        if not self._arm:
            log.warning("pick_and_place called but Robot 1 is not connected.")
            return None, None
        if self.matriz_vision is None:
            log.warning("Homography matrix not loaded.")
            return None, None

        punto_pixel = np.array([[[x_pix, y_pix]]], dtype=np.float32)
        punto_robot = cv2.perspectiveTransform(punto_pixel, self.matriz_vision)
        x_robot = float(round(punto_robot[0][0][0], 4))
        y_robot = float(round(punto_robot[0][0][1], 4))
        z_target = max(Z_GRASP_MIN, z_vision + Z_OFFSET_AGARRE)

        log.info(
            "pick_and_place → robot XY=(%.1f, %.1f) Z=%.1f",
            x_robot,
            y_robot,
            z_target,
        )

        try:
            arm = self._arm
            arm.set_position(
                x=X_RETRACT,
                y=Y_RETRACT,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=x_robot,
                y=y_robot,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=x_robot,
                y=y_robot,
                z=z_target,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=40,
                mvacc=1000,
                wait=True,
            )
            time.sleep(0.5)
            arm.close_lite6_gripper()
            time.sleep(0.8)
            arm.set_position(
                x=x_robot,
                y=y_robot,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=50,
                wait=True,
            )
            arm.set_position(
                x=X_RETRACT,
                y=Y_RETRACT,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=80,
                wait=True,
            )

            arm.set_position(
                HANDOFF_X,
                HANDOFF_Y,
                Z_HOME_SEGURO,
                HANDOFF_ROLL,
                HANDOFF_PITCH,
                HANDOFF_YAW,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                HANDOFF_X,
                HANDOFF_Y,
                HANDOFF_Z,
                HANDOFF_ROLL,
                HANDOFF_PITCH,
                HANDOFF_YAW,
                speed=40,
                mvacc=1000,
                wait=True,
            )
            time.sleep(0.5)
            arm.open_lite6_gripper()
            time.sleep(1.0)
            arm.set_position(
                HANDOFF_X,
                HANDOFF_Y,
                Z_HOME_SEGURO,
                HANDOFF_ROLL,
                HANDOFF_PITCH,
                HANDOFF_YAW,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=X_RETRACT,
                y=Y_RETRACT,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=80,
                wait=True,
            )

            log.info("Robot 1 pick_and_place complete.")
            return x_robot, y_robot
        except Exception as exc:
            log.error("Robot 1 motion error: %s", exc)
            self._recover_to_safe_home()
            return None, None

    def disconnect(self) -> None:
        if self._arm:
            try:
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None

    def _robot_init(self) -> None:
        arm = self._arm
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)
        arm.set_collision_sensitivity(3)

    def _load_calibration_matrix(self, matrix_file: str) -> None:
        try:
            self.matriz_vision = np.load(matrix_file)
            log.info("Homography matrix loaded from '%s'.", matrix_file)
        except FileNotFoundError:
            log.error("Matrix '%s' not found.", matrix_file)
            self.matriz_vision = None
        except Exception as exc:
            log.error("Matrix load error: %s", exc)
            self.matriz_vision = None

    def _recover_to_safe_home(self) -> None:
        if not self._arm:
            return
        try:
            arm = self._arm
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            arm.set_position(
                x=X_RETRACT,
                y=Y_RETRACT,
                z=Z_HOME_SEGURO,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=50,
                wait=True,
            )
            log.info("Robot 1 recovered to safe home.")
        except Exception as exc:
            log.error("Robot 1 recovery failed: %s", exc)


class RobotSorter:
    """Controls Robot 2: pickup from handoff and quality-based sorting."""

    def __init__(self, robot_ip: str):
        self._ip = robot_ip
        self._arm = None
        self._tcp_speed = 100

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    @property
    def ip(self) -> str:
        return self._ip

    def connect(self) -> bool:
        if not ROBOT_DISPONIBLE:
            log.info("Robot SDK unavailable — Robot 2 is in simulation mode.")
            return False
        try:
            self._arm = XArmAPI(self._ip, baud_checkset=False)
            self._robot_init()
            log.info("Robot 2 connected at %s.", self._ip)
            return True
        except Exception as exc:
            log.error("Robot 2 connection failed: %s", exc)
            self._arm = None
            return False

    def sort(self, quality: str) -> bool:
        """Move the handoff part to the APPROVED or REJECTED bin."""
        if not self._arm:
            log.warning("sort called but Robot 2 is not connected.")
            return False

        if quality == "APPROVED":
            dest_x, dest_y, dest_z = R2_APPROVED_X, R2_APPROVED_Y, R2_APPROVED_Z
            dest_roll = R2_APPROVED_ROLL
            dest_pitch = R2_APPROVED_PITCH
            dest_yaw = R2_APPROVED_YAW
            log.info("Robot 2 → sorting to APPROVED bin.")
        else:
            dest_x, dest_y, dest_z = R2_REJECTED_X, R2_REJECTED_Y, R2_REJECTED_Z
            dest_roll = R2_REJECTED_ROLL
            dest_pitch = R2_REJECTED_PITCH
            dest_yaw = R2_REJECTED_YAW
            log.info("Robot 2 → sorting to REJECTED bin.")

        try:
            arm = self._arm
            arm.set_position(
                x=R2_X_RETRACT,
                y=R2_Y_RETRACT,
                z=R2_Z_SAFE,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=R2_HANDOFF_X,
                y=R2_HANDOFF_Y,
                z=R2_Z_SAFE,
                roll=R2_HANDOFF_ROLL,
                pitch=R2_HANDOFF_PITCH,
                yaw=R2_HANDOFF_YAW,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=R2_HANDOFF_X,
                y=R2_HANDOFF_Y,
                z=R2_HANDOFF_Z,
                roll=R2_HANDOFF_ROLL,
                pitch=R2_HANDOFF_PITCH,
                yaw=R2_HANDOFF_YAW,
                speed=40,
                mvacc=1000,
                wait=True,
            )
            time.sleep(0.4)
            arm.close_lite6_gripper()
            time.sleep(0.8)
            arm.set_position(
                x=R2_HANDOFF_X,
                y=R2_HANDOFF_Y,
                z=R2_Z_SAFE,
                roll=R2_HANDOFF_ROLL,
                pitch=R2_HANDOFF_PITCH,
                yaw=R2_HANDOFF_YAW,
                speed=60,
                wait=True,
            )
            arm.set_position(
                x=R2_X_RETRACT,
                y=R2_Y_RETRACT,
                z=R2_Z_SAFE,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=dest_x,
                y=dest_y,
                z=R2_Z_SAFE,
                roll=dest_roll,
                pitch=dest_pitch,
                yaw=dest_yaw,
                speed=self._tcp_speed,
                wait=True,
            )
            arm.set_position(
                x=dest_x,
                y=dest_y,
                z=dest_z,
                roll=dest_roll,
                pitch=dest_pitch,
                yaw=dest_yaw,
                speed=40,
                mvacc=1000,
                wait=True,
            )
            time.sleep(0.4)
            arm.open_lite6_gripper()
            time.sleep(0.8)
            arm.set_position(
                x=dest_x,
                y=dest_y,
                z=R2_Z_SAFE,
                roll=dest_roll,
                pitch=dest_pitch,
                yaw=dest_yaw,
                speed=60,
                wait=True,
            )
            arm.set_position(
                x=R2_X_RETRACT,
                y=R2_Y_RETRACT,
                z=R2_Z_SAFE,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=self._tcp_speed,
                wait=True,
            )
            log.info("Robot 2 sort complete → %s.", quality)
            return True
        except Exception as exc:
            log.error("Robot 2 motion error: %s", exc)
            self._recover_to_safe_home()
            return False

    def disconnect(self) -> None:
        if self._arm:
            try:
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None

    def _robot_init(self) -> None:
        arm = self._arm
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(1)
        arm.set_collision_sensitivity(3)

    def _recover_to_safe_home(self) -> None:
        if not self._arm:
            return
        try:
            arm = self._arm
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            arm.set_position(
                x=R2_X_RETRACT,
                y=R2_Y_RETRACT,
                z=R2_Z_SAFE,
                roll=-179.8,
                pitch=0,
                yaw=0.1,
                speed=50,
                wait=True,
            )
            log.info("Robot 2 recovered to safe home.")
        except Exception as exc:
            log.error("Robot 2 recovery failed: %s", exc)
