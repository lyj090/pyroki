"""Mobile IK

Same as 01_basic_ik.py, but with a mobile base!
"""

import time
from collections import deque
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def _yaw_from_wxyz(wxyz: np.ndarray) -> float:
    w, x, y, z = wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def main():
    """Main function for IK with a mobile base.
    The base is fixed along the xy plane, and is biased towards being at the origin.
    """

    urdf = load_robot_description("fetch_description")
    target_link_name = "gripper_link"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0.707, 0, -0.707)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    cfg = np.array(robot.joint_var_cls(0).default_factory())
    recent_times_ms: deque[float] = deque(maxlen=100)
    iterations = 0
    target_link_index = robot.links.names.index(target_link_name)

    # Automatic stress test: toggle a large target position bias every second.
    bias_period_s = 2.0
    bias_translation = np.array([0.20, 0.20, 0.10])
    bias_enabled = False
    last_bias_toggle_time = time.time()

    # Completion timing for each bias-switch execution.
    joint_track_threshold = 5e-2
    base_pos_track_threshold_m = 2e-2
    base_yaw_track_threshold_rad = 5e-2
    execution_pending = False
    execution_start_time = 0.0
    execution_start_iteration = 0

    # Velocity limits for smooth execution (instead of instant jumps to IK solution).
    max_joint_speed = 1.0  # rad/s (and m/s for prismatic joints)
    max_base_speed = 0.40  # m/s on xy plane
    max_base_yaw_speed = 1.0  # rad/s
    last_update_time = time.time()
    last_cfg_target = cfg.copy()
    last_base_pos_target = np.array(base_frame.position)
    last_base_yaw_target = _yaw_from_wxyz(np.array(base_frame.wxyz))

    while True:
        now = time.time()
        if now - last_bias_toggle_time >= bias_period_s:
            if execution_pending:
                timeout_ms = (now - execution_start_time) * 1000
                print(
                    f"Execution timeout: time={timeout_ms:.3f} ms, solves={iterations - execution_start_iteration}"
                )
            bias_enabled = not bias_enabled
            last_bias_toggle_time = now
            execution_pending = True
            execution_start_time = now
            execution_start_iteration = iterations
            print("IK target bias: ON" if bias_enabled else "IK target bias: OFF")

        target_position = np.array(ik_target.position)
        if bias_enabled:
            target_position = target_position + bias_translation

        control_dt = max(now - last_update_time, 1e-3)
        last_update_time = now

        # Solve IK.
        start_time = time.time()
        base_pos, base_wxyz, cfg_target = pks.solve_ik_with_base(
            robot=robot,
            target_link_name=target_link_name,
            target_position=target_position,
            target_wxyz=np.array(ik_target.wxyz),
            fix_base_position=(False, False, True),  # Only free along xy plane.
            fix_base_orientation=(True, True, False),  # Free along z-axis rotation.
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz,
            prev_cfg=cfg,
        )
        base_pos_target = np.array(base_pos)
        base_wxyz_target = np.array(base_wxyz)
        last_cfg_target = cfg_target
        last_base_pos_target = base_pos_target
        last_base_yaw_target = _yaw_from_wxyz(base_wxyz_target)

        # Apply velocity limits to robot state update.
        max_joint_step = max_joint_speed * control_dt
        cfg_delta = np.clip(cfg_target - cfg, -max_joint_step, max_joint_step)
        cfg = cfg + cfg_delta

        current_base_pos = np.array(base_frame.position)
        base_delta = base_pos_target - current_base_pos
        base_delta_norm = float(np.linalg.norm(base_delta))
        max_base_step = max_base_speed * control_dt
        if base_delta_norm > max_base_step and base_delta_norm > 1e-9:
            base_delta = base_delta / base_delta_norm * max_base_step
        new_base_pos = current_base_pos + base_delta

        current_yaw = _yaw_from_wxyz(np.array(base_frame.wxyz))
        target_yaw = _yaw_from_wxyz(base_wxyz_target)
        yaw_delta = _wrap_to_pi(target_yaw - current_yaw)
        max_yaw_step = max_base_yaw_speed * control_dt
        yaw_delta = float(np.clip(yaw_delta, -max_yaw_step, max_yaw_step))
        new_yaw = current_yaw + yaw_delta
        new_base_wxyz = np.array([np.cos(new_yaw / 2.0), 0.0, 0.0, np.sin(new_yaw / 2.0)])
        # print(f"Base position: {base_pos}, Base orientation (wxyz): {base_wxyz}, Joint configuration: {cfg}")

        # Update timing handle.
        elapsed_time = time.time() - start_time
        elapsed_ms = elapsed_time * 1000
        recent_times_ms.append(elapsed_ms)
        iterations += 1
        if iterations % 100 == 0 and len(recent_times_ms) == 100:
            avg_ms = float(np.mean(recent_times_ms))
            min_ms = float(np.min(recent_times_ms))
            max_ms = float(np.max(recent_times_ms))
            print(
                f"Recent 100 IK solve times: avg={avg_ms:.3f} ms, min={min_ms:.3f} ms, max={max_ms:.3f} ms"
            )

        if execution_pending:
            cfg_track_err = float(np.max(np.abs(cfg - last_cfg_target)))
            base_pos_track_err = float(np.linalg.norm(new_base_pos - last_base_pos_target))
            base_yaw_track_err = abs(_wrap_to_pi(new_yaw - last_base_yaw_target))

            if (
                cfg_track_err <= joint_track_threshold
                and base_pos_track_err <= base_pos_track_threshold_m
                and base_yaw_track_err <= base_yaw_track_threshold_rad
            ):
                execution_time_ms = (time.time() - execution_start_time) * 1000
                execution_iters = iterations - execution_start_iteration
                print(
                    "Execution completed: "
                    f"time={execution_time_ms:.3f} ms, solves={execution_iters}, "
                    f"cfg_err={cfg_track_err:.4f}, base_pos_err={base_pos_track_err:.4f} m, "
                    f"base_yaw_err={base_yaw_track_err:.4f} rad"
                )
                execution_pending = False

        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_ms

        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = new_base_pos
        base_frame.wxyz = new_base_wxyz


if __name__ == "__main__":
    main()
