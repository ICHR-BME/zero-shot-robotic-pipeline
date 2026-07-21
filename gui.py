"""CustomTkinter industrial dashboard for FrED."""

from __future__ import annotations

import datetime
import logging
import threading
import time

import cv2
import customtkinter as ctk
from PIL import Image

import config as cfg
from database import ProductionDB
from robots import RobotMain, RobotSorter
from vision import VisionSystem

log = logging.getLogger(__name__)


class FredGUI:
    """Dashboard that connects cameras, vision, robots and traceability."""

    def __init__(
        self,
        db: ProductionDB,
        robot: RobotMain,
        robot2: RobotSorter,
        vision: VisionSystem,
        session_id: str,
    ):
        self.db = db
        self.robot = robot
        self.robot2 = robot2
        self.vision = vision
        self.session_id = session_id

        self.robot_event = threading.Event()
        self.pieza_activa = "Unknown"
        self._log_rows: list = []
        self._closed = False

        self._cam_err = False
        self._fps_t = time.time()
        self._fps_cnt = 0
        self.cap_top = cv2.VideoCapture(cfg.CAM_TOP_IDX)
        self.cap_side = cv2.VideoCapture(cfg.CAM_SIDE_IDX)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("FrED — Zero-Shot Quality, Traceability & Robotic Control")
        self.app.geometry("1380x820")
        self.app.configure(fg_color=cfg.C_BG)
        self.app.resizable(True, True)
        self.app.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_interface()
        self._tick()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════════
    def run(self) -> None:
        self._update_frame()
        log.info("GUI started.")
        self.app.mainloop()

    def release_cameras(self) -> None:
        self.cap_top.release()
        self.cap_side.release()

    # ══════════════════════════════════════════════════════════════════════════
    # GUI HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _schedule(self, delay_ms: int, callback) -> None:
        if self._closed:
            return
        try:
            self.app.after(delay_ms, callback)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.app.destroy()
        except Exception:
            pass

    def _card(self, parent, **kwargs):
        options = {
            "fg_color": cfg.C_SURFACE,
            "corner_radius": 8,
            "border_width": 1,
            "border_color": cfg.C_BORDER,
        }
        options.update(kwargs)
        return ctk.CTkFrame(parent, **options)

    def _section_header(self, parent, text: str) -> None:
        ctk.CTkLabel(
            parent,
            text="▌ " + text.upper(),
            text_color=cfg.C_ACCENT,
            font=cfg.FONT_SMALL,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(10, 4))

    def _icon_btn(
        self,
        parent,
        text,
        command,
        fg,
        hover,
        text_color,
        border,
        height=34,
        font=cfg.FONT_SMALL,
    ):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            fg_color=fg,
            hover_color=hover,
            text_color=text_color,
            border_width=1,
            border_color=border,
            font=font,
            corner_radius=6,
            height=height,
        )

    def _stat_box(self, parent, label, value, color):
        frame = ctk.CTkFrame(
            parent,
            fg_color=cfg.C_SURFACE2,
            corner_radius=6,
            border_width=1,
            border_color=cfg.C_BORDER,
        )
        frame.pack(side="left", expand=True, fill="both", padx=3)
        ctk.CTkLabel(
            frame,
            text=label,
            text_color=cfg.C_GRAY,
            font=("Consolas", 9),
        ).pack(pady=(6, 0))
        value_label = ctk.CTkLabel(
            frame,
            text=value,
            text_color=color,
            font=("Consolas", 16, "bold"),
        )
        value_label.pack(pady=(0, 6))
        return value_label

    def _slider_row(self, parent, label, start, end, steps, default, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))

        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(
            top,
            text=label,
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
            anchor="w",
        ).pack(side="left")
        value_label = ctk.CTkLabel(
            top,
            text=f"{default:.0f}",
            text_color=color,
            font=("Consolas", 10),
            anchor="e",
        )
        value_label.pack(side="right")

        slider = ctk.CTkSlider(
            row,
            from_=start,
            to=end,
            number_of_steps=steps,
            fg_color=cfg.C_SURFACE2,
            progress_color=color,
            button_color=color,
            button_hover_color=cfg.C_TEXT,
            command=lambda value: value_label.configure(text=f"{value:.0f}"),
        )
        slider.set(default)
        slider.pack(fill="x", pady=(2, 0))
        return slider

    def _coord_row(self, parent, label, value, color=cfg.C_GRAY):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(
            frame,
            text=label,
            text_color=cfg.C_GRAY,
            font=("Consolas", 9),
            anchor="w",
            width=120,
        ).pack(side="left")
        ctk.CTkLabel(
            frame,
            text=value,
            text_color=color,
            font=("Consolas", 9),
            anchor="w",
        ).pack(side="left")

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFACE CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════
    def _build_interface(self) -> None:
        topbar = ctk.CTkFrame(
            self.app,
            fg_color=cfg.C_SURFACE,
            height=44,
            corner_radius=0,
        )
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        ctk.CTkLabel(
            topbar,
            text="  ◈  FrED VISION SYSTEM",
            text_color=cfg.C_ACCENT,
            font=("Consolas", 14, "bold"),
        ).pack(side="left", padx=14)
        ctk.CTkLabel(
            topbar,
            text="GROUNDED-SAM  ·  DINOv2  ·  xARM  ·  ZERO-SHOT",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
        ).pack(side="left", padx=6)

        self.lbl_clock = ctk.CTkLabel(
            topbar,
            text="",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
        )
        self.lbl_clock.pack(side="right", padx=16)
        ctk.CTkLabel(
            topbar,
            text=f"SESSION  {self.session_id}",
            text_color=cfg.C_ACCENT2,
            font=("Consolas", 10),
        ).pack(side="right", padx=16)

        body = ctk.CTkFrame(self.app, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0, minsize=360)
        body.rowconfigure(0, weight=1)

        self._build_left_column(body)
        self._build_right_column(body)

    def _build_left_column(self, body) -> None:
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=0)
        left.columnconfigure(0, weight=1)

        cam_card = self._card(left)
        cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        cam_header = ctk.CTkFrame(
            cam_card,
            fg_color=cfg.C_SURFACE2,
            corner_radius=6,
            height=32,
        )
        cam_header.pack(fill="x", padx=8, pady=(8, 4))
        cam_header.pack_propagate(False)
        ctk.CTkLabel(
            cam_header,
            text="  TOP CAMERA  —  GROUNDED-SAM + DINOv2",
            text_color=cfg.C_ACCENT,
            font=("Consolas", 10),
        ).pack(side="left", padx=8)
        self.lbl_fps = ctk.CTkLabel(
            cam_header,
            text="FPS: --",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
        )
        self.lbl_fps.pack(side="right", padx=10)

        self.lbl_video = ctk.CTkLabel(
            cam_card,
            text="⚠  Camera initialising…",
            text_color=cfg.C_GRAY,
            font=("Consolas", 12),
            width=cfg.CAM_W,
            height=cfg.CAM_H,
            fg_color=cfg.C_BG,
            corner_radius=4,
        )
        self.lbl_video.pack(padx=8, pady=(0, 8))

        detection_card = self._card(left, height=80)
        detection_card.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        detection_card.pack_propagate(False)

        detection_inner = ctk.CTkFrame(
            detection_card,
            fg_color="transparent",
        )
        detection_inner.pack(fill="both", expand=True, padx=14, pady=8)

        self.lbl_part_name = ctk.CTkLabel(
            detection_inner,
            text="AWAITING DETECTION",
            text_color=cfg.C_ACCENT,
            font=("Consolas", 18, "bold"),
            anchor="w",
        )
        self.lbl_part_name.pack(fill="x")
        self.lbl_quality_status = ctk.CTkLabel(
            detection_inner,
            text="─   No part in frame",
            text_color=cfg.C_GRAY,
            font=("Consolas", 12),
            anchor="w",
        )
        self.lbl_quality_status.pack(fill="x")

        stats_card = self._card(left, height=66)
        stats_card.grid(row=2, column=0, sticky="ew")
        stats_card.pack_propagate(False)
        stats_inner = ctk.CTkFrame(stats_card, fg_color="transparent")
        stats_inner.pack(fill="both", expand=True, padx=8, pady=8)

        totals = self.db.get_totals()
        self.lbl_stat_total = self._stat_box(
            stats_inner,
            "TOTAL PICKED",
            str(totals["total"]),
            cfg.C_TEXT,
        )
        self.lbl_stat_approved = self._stat_box(
            stats_inner,
            "APPROVED",
            str(totals["approved"]),
            cfg.C_GREEN,
        )
        self.lbl_stat_rejected = self._stat_box(
            stats_inner,
            "REJECTED",
            str(totals["rejected"]),
            cfg.C_RED,
        )
        self.lbl_stat_session = self._stat_box(
            stats_inner,
            "THIS SESSION",
            "0",
            cfg.C_ACCENT2,
        )

    def _build_right_column(self, body) -> None:
        right_outer = self._card(body)
        right_outer.grid(row=0, column=1, sticky="nsew")

        right_scroll = ctk.CTkScrollableFrame(
            right_outer,
            fg_color="transparent",
            scrollbar_button_color=cfg.C_BORDER,
            scrollbar_button_hover_color=cfg.C_ACCENT,
            width=340,
        )
        right_scroll.pack(fill="both", expand=True, padx=2, pady=2)

        self._build_registration_card(right_scroll)
        self._build_threshold_card(right_scroll)
        self._build_robot_card(right_scroll)
        self._build_log_card(right_scroll)

    def _build_registration_card(self, parent) -> None:
        registration_card = self._card(parent)
        registration_card.pack(fill="x", pady=(0, 6))
        self._section_header(registration_card, "Part Registration")

        self.entry_pieza = ctk.CTkEntry(
            registration_card,
            placeholder_text="Enter part name…",
            fg_color=cfg.C_SURFACE2,
            border_color=cfg.C_BORDER,
            text_color=cfg.C_TEXT,
            font=("Consolas", 11),
            corner_radius=6,
            height=34,
        )
        self.entry_pieza.pack(padx=10, pady=(0, 6), fill="x")

        self.lbl_active_part = ctk.CTkLabel(
            registration_card,
            text="Active:  —",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
            anchor="w",
        )
        self.lbl_active_part.pack(padx=12, fill="x")

        define_button = self._icon_btn(
            registration_card,
            "① DEFINE PART",
            self._set_active_part,
            cfg.C_SURFACE2,
            cfg.C_BORDER,
            cfg.C_TEXT,
            cfg.C_ACCENT,
        )
        define_button.pack(padx=10, pady=(6, 4), fill="x")

        self.lbl_capture_info = ctk.CTkLabel(
            registration_card,
            text="0 angles captured",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
            anchor="w",
        )
        self.lbl_capture_info.pack(padx=12, fill="x")

        self.btn_capture = self._icon_btn(
            registration_card,
            "② CAPTURE ANGLE  (0)",
            self._capture_angle,
            "#0D260D",
            "#123012",
            cfg.C_GREEN,
            cfg.C_GREEN,
        )
        self.btn_capture.pack(padx=10, pady=(4, 12), fill="x")

    def _build_threshold_card(self, parent) -> None:
        threshold_card = self._card(parent)
        threshold_card.pack(fill="x", pady=(0, 6))
        self._section_header(threshold_card, "Detection Thresholds")

        self.slider_identidad = self._slider_row(
            threshold_card,
            "③ IDENTITY TOLERANCE",
            0,
            1000,
            250,
            cfg.DEFAULT_IDENTITY_TOLERANCE,
            cfg.C_ACCENT,
        )
        self.slider_calidad = self._slider_row(
            threshold_card,
            "④ QUALITY TOLERANCE",
            0,
            1000,
            150,
            cfg.DEFAULT_QUALITY_TOLERANCE,
            cfg.C_GREEN,
        )

    def _build_robot_card(self, parent) -> None:
        robot_card = self._card(parent)
        robot_card.pack(fill="x", pady=(0, 6))
        self._section_header(robot_card, "Robot Control")

        self.lbl_robot_status = ctk.CTkLabel(
            robot_card,
            text="●  R1 DISCONNECTED",
            text_color=cfg.C_RED,
            font=("Consolas", 10),
            anchor="w",
        )
        self.lbl_robot_status.pack(padx=12, pady=(0, 4), fill="x")

        self.lbl_robot2_status = ctk.CTkLabel(
            robot_card,
            text="●  R2 DISCONNECTED",
            text_color=cfg.C_RED,
            font=("Consolas", 10),
            anchor="w",
        )
        self.lbl_robot2_status.pack(padx=12, pady=(0, 6), fill="x")

        self.btn_conectar = self._icon_btn(
            robot_card,
            "CONNECT  ROBOT 1  (Picker)",
            self._connect_robot1,
            cfg.C_SURFACE2,
            cfg.C_BORDER,
            cfg.C_TEXT,
            cfg.C_ACCENT2,
        )
        self.btn_conectar.pack(padx=10, pady=(0, 4), fill="x")

        self.btn_conectar2 = self._icon_btn(
            robot_card,
            "CONNECT  ROBOT 2  (Sorter)",
            self._connect_robot2,
            cfg.C_SURFACE2,
            cfg.C_BORDER,
            cfg.C_TEXT,
            cfg.C_ACCENT2,
        )
        self.btn_conectar2.pack(padx=10, pady=(0, 6), fill="x")

        coordinates = ctk.CTkFrame(
            robot_card,
            fg_color=cfg.C_SURFACE2,
            corner_radius=6,
        )
        coordinates.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(
            coordinates,
            text="HANDOFF POINT  (R1 drops / R2 picks)",
            text_color=cfg.C_ACCENT,
            font=("Consolas", 9),
        ).pack(padx=8, pady=(6, 2), anchor="w")
        self._coord_row(
            coordinates,
            "R1 drop:",
            f"X={cfg.HANDOFF_X:.0f}  Y={cfg.HANDOFF_Y:.0f}  Z={cfg.HANDOFF_Z:.0f}",
            cfg.C_TEXT,
        )
        self._coord_row(
            coordinates,
            "R2 pickup:",
            f"X={cfg.R2_HANDOFF_X:.0f}  Y={cfg.R2_HANDOFF_Y:.0f}  Z={cfg.R2_HANDOFF_Z:.0f}",
            cfg.C_TEXT,
        )

        ctk.CTkLabel(
            coordinates,
            text="SORT DESTINATIONS",
            text_color=cfg.C_ACCENT,
            font=("Consolas", 9),
        ).pack(padx=8, pady=(6, 2), anchor="w")
        self._coord_row(
            coordinates,
            "✓ APPROVED:",
            f"X={cfg.R2_APPROVED_X:.0f}  Y={cfg.R2_APPROVED_Y:.0f}  Z={cfg.R2_APPROVED_Z:.0f}",
            cfg.C_GREEN,
        )
        self._coord_row(
            coordinates,
            "✗ REJECTED:",
            f"X={cfg.R2_REJECTED_X:.0f}  Y={cfg.R2_REJECTED_Y:.0f}  Z={cfg.R2_REJECTED_Z:.0f}",
            cfg.C_RED,
        )
        ctk.CTkLabel(
            coordinates,
            text="  ⚠  Verify all coordinates before automatic operation",
            text_color=cfg.C_YELLOW,
            font=("Consolas", 8),
        ).pack(padx=8, pady=(2, 6), anchor="w")

        self.lbl_exec_status = ctk.CTkLabel(
            robot_card,
            text="─   Idle",
            text_color=cfg.C_GRAY,
            font=("Consolas", 10),
            anchor="w",
        )
        self.lbl_exec_status.pack(padx=12, pady=(0, 4), fill="x")

        self.btn_robot = self._icon_btn(
            robot_card,
            "⑤  EXECUTE PICK & PLACE",
            self._launch_robot_sequence,
            "#0A1A2E",
            "#0F2540",
            cfg.C_ACCENT,
            cfg.C_ACCENT,
            height=44,
            font=("Consolas", 13, "bold"),
        )
        self.btn_robot.configure(state="disabled")
        self.btn_robot.pack(padx=10, pady=(0, 6), fill="x")

        self.btn_export = self._icon_btn(
            robot_card,
            "EXPORT CSV",
            self._export_csv,
            cfg.C_SURFACE2,
            cfg.C_BORDER,
            cfg.C_TEXT,
            cfg.C_BORDER,
        )
        self.btn_export.pack(padx=10, pady=(0, 12), fill="x")

    def _build_log_card(self, parent) -> None:
        log_card = self._card(parent)
        log_card.pack(fill="x", pady=(0, 4))
        self._section_header(log_card, "Production Log")

        header = ctk.CTkFrame(
            log_card,
            fg_color=cfg.C_SURFACE2,
            corner_radius=0,
        )
        header.pack(fill="x", padx=8, pady=(0, 2))
        for column, width in [
            ("#", 32),
            ("PART", 90),
            ("QTY", 68),
            ("DIST", 52),
            ("TIME", 72),
        ]:
            ctk.CTkLabel(
                header,
                text=column,
                text_color=cfg.C_GRAY,
                font=("Consolas", 9),
                width=width,
                anchor="w",
            ).pack(side="left", padx=3, pady=3)

        self.log_scroll_inner = ctk.CTkScrollableFrame(
            log_card,
            fg_color="transparent",
            height=220,
            scrollbar_button_color=cfg.C_BORDER,
            scrollbar_button_hover_color=cfg.C_ACCENT,
        )
        self.log_scroll_inner.pack(fill="x", padx=8, pady=(0, 8))

        for row in reversed(self.db.get_recent(20)):
            self._append_log_row(
                row["id"],
                row["part_name"],
                row["quality"],
                row["distance"],
            )

    # ══════════════════════════════════════════════════════════════════════════
    # CLOCK, STATS AND PART REGISTRATION
    # ══════════════════════════════════════════════════════════════════════════
    def _tick(self) -> None:
        if self._closed:
            return
        self.lbl_clock.configure(
            text=datetime.datetime.now().strftime("%Y-%m-%d   %H:%M:%S   ")
        )
        self._schedule(1000, self._tick)

    def _refresh_stats(self) -> None:
        totals = self.db.get_totals()
        session = self.db.get_session_stats(self.session_id)
        self.lbl_stat_total.configure(text=str(totals["total"]))
        self.lbl_stat_approved.configure(text=str(totals["approved"]))
        self.lbl_stat_rejected.configure(text=str(totals["rejected"]))
        self.lbl_stat_session.configure(text=str(session.get("total_parts", 0)))

    def _set_active_part(self) -> None:
        name = self.entry_pieza.get().strip()
        if name:
            self.pieza_activa = name
            self.lbl_active_part.configure(
                text=f"Active:  {self.pieza_activa}",
                text_color=cfg.C_ACCENT,
            )
        else:
            self.lbl_active_part.configure(
                text="⚠  Enter a name first.",
                text_color=cfg.C_YELLOW,
            )

    def _capture_angle(self) -> None:
        if self.pieza_activa == "Unknown":
            self.lbl_capture_info.configure(
                text="⚠  Define a part name first.",
                text_color=cfg.C_YELLOW,
            )
            return

        try:
            count = self.vision.register_sample(self.pieza_activa)
        except RuntimeError:
            self.lbl_capture_info.configure(
                text="⚠  Waiting for SAM detection…",
                text_color=cfg.C_YELLOW,
            )
            return
        except Exception as exc:
            log.error("Part sample registration failed: %s", exc)
            self.lbl_capture_info.configure(
                text="✘  Could not capture sample.",
                text_color=cfg.C_RED,
            )
            return

        self.lbl_capture_info.configure(
            text=f"✔  {count} angle(s) for '{self.pieza_activa}'",
            text_color=cfg.C_GREEN,
        )
        self.btn_capture.configure(text=f"② CAPTURE ANGLE  ({count})")

    # ══════════════════════════════════════════════════════════════════════════
    # ROBOT CONNECTION AND EXECUTION
    # ══════════════════════════════════════════════════════════════════════════
    def _connect_robot1(self) -> None:
        threading.Thread(
            target=self._connect_robot1_worker,
            name="connect-robot-1",
            daemon=True,
        ).start()

    def _connect_robot1_worker(self) -> None:
        self._schedule(
            0,
            lambda: self.btn_conectar.configure(
                text="R1 CONNECTING…",
                state="disabled",
                fg_color=cfg.C_SURFACE2,
                text_color=cfg.C_YELLOW,
                border_color=cfg.C_YELLOW,
            ),
        )
        success = self.robot.connect()

        if success:
            def connected_ui():
                self.lbl_robot_status.configure(
                    text=f"●  R1 CONNECTED  ({self.robot.ip})",
                    text_color=cfg.C_GREEN,
                )
                self.btn_conectar.configure(
                    text="✔  R1 CONNECTED",
                    fg_color="#0D260D",
                    text_color=cfg.C_GREEN,
                    border_color=cfg.C_GREEN,
                    state="disabled",
                )
                self._check_both_connected()

            self._schedule(0, connected_ui)
        else:
            def failed_ui():
                self.lbl_robot_status.configure(
                    text="●  R1 CONNECTION FAILED",
                    text_color=cfg.C_RED,
                )
                self.btn_conectar.configure(
                    text="RETRY R1",
                    state="normal",
                    fg_color=cfg.C_SURFACE2,
                    text_color=cfg.C_RED,
                    border_color=cfg.C_RED,
                )

            self._schedule(0, failed_ui)

    def _connect_robot2(self) -> None:
        threading.Thread(
            target=self._connect_robot2_worker,
            name="connect-robot-2",
            daemon=True,
        ).start()

    def _connect_robot2_worker(self) -> None:
        self._schedule(
            0,
            lambda: self.btn_conectar2.configure(
                text="R2 CONNECTING…",
                state="disabled",
                fg_color=cfg.C_SURFACE2,
                text_color=cfg.C_YELLOW,
                border_color=cfg.C_YELLOW,
            ),
        )
        success = self.robot2.connect()

        if success:
            def connected_ui():
                self.lbl_robot2_status.configure(
                    text=f"●  R2 CONNECTED  ({self.robot2.ip})",
                    text_color=cfg.C_GREEN,
                )
                self.btn_conectar2.configure(
                    text="✔  R2 CONNECTED",
                    fg_color="#0D260D",
                    text_color=cfg.C_GREEN,
                    border_color=cfg.C_GREEN,
                    state="disabled",
                )
                self._check_both_connected()

            self._schedule(0, connected_ui)
        else:
            def failed_ui():
                self.lbl_robot2_status.configure(
                    text="●  R2 CONNECTION FAILED",
                    text_color=cfg.C_RED,
                )
                self.btn_conectar2.configure(
                    text="RETRY R2",
                    state="normal",
                    fg_color=cfg.C_SURFACE2,
                    text_color=cfg.C_RED,
                    border_color=cfg.C_RED,
                )

            self._schedule(0, failed_ui)

    def _check_both_connected(self) -> None:
        if self.robot.is_connected and self.robot2.is_connected:
            self.btn_robot.configure(state="normal")

    def _launch_robot_sequence(self) -> None:
        if self.robot_event.is_set():
            return

        self.robot_event.set()
        snapshot = self.vision.snapshot()
        quality_threshold = float(self.slider_calidad.get())
        threading.Thread(
            target=self._execute_robot_sequence,
            args=(snapshot, quality_threshold),
            name="robot-sequence",
            daemon=True,
        ).start()

    def _set_ui_busy(self, message: str) -> None:
        self.btn_robot.configure(
            text=message,
            state="disabled",
            fg_color="#261A00",
            text_color=cfg.C_YELLOW,
            border_color=cfg.C_YELLOW,
        )
        self.lbl_exec_status.configure(
            text=f"⏳  {message}",
            text_color=cfg.C_YELLOW,
        )

    def _set_ui_idle(self) -> None:
        state = (
            "normal"
            if self.robot.is_connected and self.robot2.is_connected
            else "disabled"
        )
        self.btn_robot.configure(
            text="⑤  EXECUTE PICK & PLACE",
            state=state,
            fg_color="#0A1A2E",
            text_color=cfg.C_ACCENT,
            border_color=cfg.C_ACCENT,
        )

    def _execute_robot_sequence(self, snapshot: dict, quality_threshold: float) -> None:
        try:
            part_name = snapshot["pieza"]
            distance = snapshot["distancia"]
            x_pix = snapshot["x"]
            y_pix = snapshot["y"]
            z_mm = snapshot["z"]
            quality = "APPROVED" if distance <= quality_threshold else "REJECTED"

            self._schedule(0, lambda: self._set_ui_busy("R1 picking from table…"))
            x_robot, y_robot = self.robot.pick_and_place(x_pix, y_pix, z_mm)

            if x_robot is None:
                log.error("Robot 1 pick failed — aborting sequence.")

                def r1_failed_ui():
                    self.lbl_exec_status.configure(
                        text="✘  R1 pick failed",
                        text_color=cfg.C_RED,
                    )
                    self._set_ui_idle()

                self._schedule(0, r1_failed_ui)
                return

            self._schedule(
                0,
                lambda: self._set_ui_busy(f"R2 sorting → {quality}…"),
            )
            sort_ok = self.robot2.sort(quality)
            if not sort_ok:
                log.error("Robot 2 sort failed.")
                self._schedule(
                    0,
                    lambda: self.lbl_exec_status.configure(
                        text="✘  R2 sort failed",
                        text_color=cfg.C_RED,
                    ),
                )

            row_id = self.db.log_pick(
                session_id=self.session_id,
                part_name=part_name,
                quality=quality,
                distance=distance,
                x_pix=x_pix,
                y_pix=y_pix,
                z_mm=z_mm,
                x_robot=x_robot,
                y_robot=y_robot,
            )

            sort_label = "✓ APPROVED" if quality == "APPROVED" else "✗ REJECTED"

            def completed_ui():
                self._refresh_stats()
                self._append_log_row(
                    row_id,
                    part_name,
                    quality,
                    distance,
                )
                self.lbl_exec_status.configure(
                    text=f"Last: {part_name}  →  {sort_label}",
                    text_color=(
                        cfg.C_GREEN if quality == "APPROVED" else cfg.C_RED
                    ),
                )
                self._set_ui_idle()

            self._schedule(0, completed_ui)
        except Exception as exc:
            log.error("Robot sequence failed: %s", exc, exc_info=True)

            def unexpected_failure_ui():
                self.lbl_exec_status.configure(
                    text="✘  Unexpected execution error",
                    text_color=cfg.C_RED,
                )
                self._set_ui_idle()

            self._schedule(0, unexpected_failure_ui)
        finally:
            self.robot_event.clear()

    def _export_csv(self) -> None:
        path = f"export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            self.db.export_csv(path)
            self.btn_export.configure(text=f"✔  {path}", text_color=cfg.C_GREEN)
            self._schedule(
                3000,
                lambda: self.btn_export.configure(
                    text="EXPORT CSV",
                    text_color=cfg.C_TEXT,
                ),
            )
        except Exception as exc:
            log.error("CSV export failed: %s", exc)
            self.btn_export.configure(text="✘  EXPORT FAILED", text_color=cfg.C_RED)

    def _append_log_row(
        self,
        row_id: int,
        part: str,
        quality: str,
        distance: float,
    ) -> None:
        color = cfg.C_GREEN if quality == "APPROVED" else cfg.C_RED
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        row_frame = ctk.CTkFrame(
            self.log_scroll_inner,
            fg_color=cfg.C_SURFACE2,
            corner_radius=3,
            height=24,
        )
        row_frame.pack(fill="x", pady=1)
        row_frame.pack_propagate(False)

        for value, width, text_color in [
            (f"#{row_id}", 32, cfg.C_GRAY),
            (part[:10], 90, cfg.C_TEXT),
            (quality[:3], 68, color),
            (f"{distance:.0f}", 52, cfg.C_GRAY),
            (timestamp, 72, cfg.C_GRAY),
        ]:
            ctk.CTkLabel(
                row_frame,
                text=value,
                text_color=text_color,
                font=("Consolas", 9),
                width=width,
                anchor="w",
            ).pack(side="left", padx=3)

        self._log_rows.append(row_frame)
        if len(self._log_rows) > 100:
            self._log_rows.pop(0).destroy()

    # ══════════════════════════════════════════════════════════════════════════
    # CAMERA RENDER LOOP
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _try_reopen(capture, index: int) -> bool:
        capture.release()
        capture.open(index)
        return capture.isOpened()

    def _update_frame(self) -> None:
        if self._closed:
            return

        ret_top, frame = self.cap_top.read()
        ret_side, side_frame = self.cap_side.read()

        self._fps_cnt += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.lbl_fps.configure(text=f"FPS: {self._fps_cnt}")
            self._fps_cnt = 0
            self._fps_t = now

        if not ret_top:
            if not self._cam_err:
                log.warning("Top camera lost — reconnecting.")
                self._cam_err = True
            self.lbl_video.configure(image=None, text="⚠  Top camera unavailable")
            self._try_reopen(self.cap_top, cfg.CAM_TOP_IDX)
            self._schedule(500, self._update_frame)
            return

        self._cam_err = False
        self.vision.set_frames(frame, side_frame if ret_side else None)

        snapshot = self.vision.snapshot()
        x = int(snapshot["x"])
        y = int(snapshot["y"])
        z_mm = float(snapshot["z"])
        mask = snapshot["mask_top"]
        distance = float(snapshot["distancia"])
        part_name = snapshot["pieza"]
        roi = snapshot["roi_actual"]
        identity_threshold = float(self.slider_identidad.get())
        quality_threshold = float(self.slider_calidad.get())

        if mask is not None:
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(frame, contours, -1, (0, 210, 180), 2)

        cv2.drawMarker(
            frame,
            (x, y),
            (0, 80, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )

        cv2.rectangle(frame, (4, 4), (210, 64), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"XY: {x}, {y}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 200, 255),
            1,
        )
        cv2.putText(
            frame,
            f" Z: {z_mm:.1f} mm",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 230, 160),
            1,
        )

        half = cfg.ROI_SIZE // 2
        x1, y1, x2, y2 = x - half, y - half, x + half, y + half

        if self.vision.sample_count > 0 and roi is not None:
            if distance > identity_threshold:
                self.lbl_part_name.configure(
                    text="UNKNOWN OBJECT",
                    text_color=cfg.C_GRAY,
                )
                self.lbl_quality_status.configure(
                    text=f"─   Out of catalogue  (dist {distance:.0f})",
                    text_color=cfg.C_GRAY,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 3)
                self.btn_robot.configure(state="disabled")
            else:
                self.lbl_part_name.configure(
                    text=part_name.upper(),
                    text_color=cfg.C_ACCENT,
                )
                if (
                    not self.robot_event.is_set()
                    and self.robot.is_connected
                    and self.robot2.is_connected
                ):
                    self.btn_robot.configure(state="normal")

                if distance > quality_threshold:
                    self.lbl_quality_status.configure(
                        text=f"✗   QUALITY REJECTED  —  dist {distance:.0f}",
                        text_color=cfg.C_RED,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 220), 3)
                else:
                    self.lbl_quality_status.configure(
                        text=f"✓   QUALITY APPROVED  —  dist {distance:.0f}",
                        text_color=cfg.C_GREEN,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 3)
        else:
            self.lbl_part_name.configure(
                text="AWAITING DETECTION",
                text_color=cfg.C_ACCENT,
            )
            self.lbl_quality_status.configure(
                text="─   No part in frame",
                text_color=cfg.C_GRAY,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 140, 0), 2)
            self.btn_robot.configure(state="disabled")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        image_tk = ctk.CTkImage(
            light_image=image_pil,
            dark_image=image_pil,
            size=(cfg.CAM_W, cfg.CAM_H),
        )
        self.lbl_video.configure(image=image_tk, text="")
        self.lbl_video.image = image_tk

        self._schedule(15, self._update_frame)
