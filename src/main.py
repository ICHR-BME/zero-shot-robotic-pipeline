"""Entry point: connects the FrED GUI, AI, database and robots."""

from __future__ import annotations

import logging

from config import CALIB_MATRIX_FILE, DB_FILE, ROBOT2_IP, ROBOT_IP
from database import ProductionDB
from gui import FredGUI
from robots import RobotMain, RobotSorter
from vision import VisionSystem

log = logging.getLogger(__name__)


def main() -> None:
    db = ProductionDB(DB_FILE)
    robot = RobotMain(ROBOT_IP, CALIB_MATRIX_FILE)
    robot2 = RobotSorter(ROBOT2_IP)
    vision = VisionSystem()
    session_id = db.start_session()
    gui = None

    try:
        vision.start()
        gui = FredGUI(
            db=db,
            robot=robot,
            robot2=robot2,
            vision=vision,
            session_id=session_id,
        )
        gui.run()
    except KeyboardInterrupt:
        log.info("Application interrupted by user.")
    except Exception as exc:
        log.error("Fatal application error: %s", exc, exc_info=True)
        raise
    finally:
        db.close_session(session_id)
        if gui is not None:
            gui.release_cameras()
        vision.stop()
        robot.disconnect()
        robot2.disconnect()
        log.info("Resources released. Goodbye.")


if __name__ == "__main__":
    main()
