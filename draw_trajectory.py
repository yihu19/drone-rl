"""
Draw a 2D drone trajectory on a canvas, then fly it in ProjectAirSim.

Controls (Drawing phase)
------------------------
  Left-drag    Draw freehand path
  Right-click  Clear path
  Space/Enter  Execute path in simulator
  Esc          Quit without flying

The canvas centre maps to the spawn point (-1, 8, *) in NED.
Canvas right → East (+Y), Canvas up → North (+X).
Scale: 1 pixel = SCALE metres.

The drone flies each waypoint at constant FLY_Z (NED, negative = above ground).
"""

import asyncio
import math
import os
import sys

import cv2
import numpy as np
import projectairsim
from projectairsim.types import Pose, Vector3, Quaternion

# ── Config ────────────────────────────────────────────────────────────────────
_SIM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "client", "python", "example_user_scripts", "sim_config",
)
_SCENE_NAME = "scene_basic_drone.jsonc"

_yaw = math.radians(-45)
_SPAWN_POSE = Pose({
    "translation": Vector3({"x": -1.0, "y": 8.0, "z": -4.0}),
    "rotation": Quaternion({"w": math.cos(_yaw / 2), "x": 0.0,
                            "y": 0.0, "z": math.sin(_yaw / 2)}),
})

SPAWN_NED   = np.array([-1.0, 8.0])   # (north, east) of spawn
FLY_Z       = -8.0                     # constant altitude (NED, negative = up)
FLY_SPEED   = 3.0                      # m/s between waypoints
SCALE       = 0.1                      # metres per pixel
MIN_WP_DIST = 0.5                      # min distance (m) between waypoints
CANVAS_W    = 800
CANVAS_H    = 800

# ── Coordinate helpers ────────────────────────────────────────────────────────
def canvas_to_ned(px, py):
    """Canvas pixel → NED (north, east)."""
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    north = SPAWN_NED[0] + (cy - py) * SCALE   # canvas-up = NED-north
    east  = SPAWN_NED[1] + (px - cx) * SCALE   # canvas-right = NED-east
    return north, east

def ned_to_canvas(north, east):
    """NED (north, east) → canvas pixel."""
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    px = int(cx + (east  - SPAWN_NED[1]) / SCALE)
    py = int(cy - (north - SPAWN_NED[0]) / SCALE)
    return px, py

# ── Drawing state ─────────────────────────────────────────────────────────────
_raw_pts  = []   # list of (px, py) canvas pixels drawn so far
_drawing  = False

def _make_canvas():
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Grid lines every 50 px (= 5 m)
    for x in range(0, CANVAS_W, 50):
        cv2.line(canvas, (x, 0), (x, CANVAS_H - 1), (40, 40, 40), 1)
    for y in range(0, CANVAS_H, 50):
        cv2.line(canvas, (0, y), (CANVAS_W - 1, y), (40, 40, 40), 1)

    # Axes through centre
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    cv2.line(canvas, (cx, 0), (cx, CANVAS_H - 1), (60, 60, 60), 1)
    cv2.line(canvas, (0, cy), (CANVAS_W - 1, cy), (60, 60, 60), 1)

    # Labels
    cv2.putText(canvas, "N", (cx + 4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
    cv2.putText(canvas, "S", (cx + 4, CANVAS_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
    cv2.putText(canvas, "E", (CANVAS_W - 18, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)
    cv2.putText(canvas, "W", (4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

    # Scale bar: 10 m = 100 px
    bar_x, bar_y = 20, CANVAS_H - 20
    cv2.line(canvas, (bar_x, bar_y), (bar_x + 100, bar_y), (160, 160, 160), 2)
    cv2.putText(canvas, "10 m", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Spawn marker
    sx, sy = ned_to_canvas(*SPAWN_NED)
    cv2.drawMarker(canvas, (sx, sy), (0, 255, 255),
                   cv2.MARKER_CROSS, 16, 2)
    cv2.putText(canvas, "SPAWN", (sx + 8, sy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return canvas

def _redraw(canvas_base):
    canvas = canvas_base.copy()
    if len(_raw_pts) > 1:
        pts = np.array(_raw_pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 140, 255), 2)
        # Draw first and last point markers
        cv2.circle(canvas, tuple(_raw_pts[0]),  5, (0, 255, 0),  -1)
        cv2.circle(canvas, tuple(_raw_pts[-1]), 5, (0, 0, 255),  -1)
    cv2.putText(canvas,
                "Left-drag: draw | Right: clear | Space/Enter: fly | Esc: quit",
                (6, CANVAS_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 180, 180), 1)
    return canvas

def _mouse_cb(event, x, y, flags, param):
    global _drawing, _raw_pts
    canvas_base = param

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _raw_pts = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            _raw_pts.append((x, y))
        cv2.imshow("Draw Trajectory", _redraw(canvas_base))
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _raw_pts.append((x, y))
        cv2.imshow("Draw Trajectory", _redraw(canvas_base))
    elif event == cv2.EVENT_RBUTTONDOWN:
        _raw_pts = []
        cv2.imshow("Draw Trajectory", _redraw(canvas_base))

# ── Waypoint extraction ───────────────────────────────────────────────────────
def extract_waypoints():
    """Downsample _raw_pts to NED waypoints spaced >= MIN_WP_DIST apart."""
    if len(_raw_pts) < 2:
        return []
    wps = []
    last = canvas_to_ned(*_raw_pts[0])
    wps.append(last)
    for pt in _raw_pts[1:]:
        ned = canvas_to_ned(*pt)
        if math.hypot(ned[0] - last[0], ned[1] - last[1]) >= MIN_WP_DIST:
            wps.append(ned)
            last = ned
    # Always include the final drawn point
    final = canvas_to_ned(*_raw_pts[-1])
    if math.hypot(final[0] - last[0], final[1] - last[1]) >= MIN_WP_DIST / 4:
        wps.append(final)
    return wps

# ── Simulator execution ───────────────────────────────────────────────────────
async def fly_trajectory(waypoints):
    print(f"\nConnecting to simulator…  ({len(waypoints)} waypoints)")

    client = projectairsim.ProjectAirSimClient()
    client.connect()
    world  = projectairsim.World(client, _SCENE_NAME,
                                 sim_config_path=_SIM_CONFIG_PATH,
                                 delay_after_load_sec=2)
    drone  = projectairsim.Drone(client, world, "Drone1")

    # Reset to spawn
    try:
        drone.disarm()
    except Exception:
        pass
    drone.set_pose(_SPAWN_POSE, True)
    drone.enable_api_control()
    drone.arm()
    print("Taking off…")
    await (await drone.takeoff_async())
    print(f"Airborne — {len(waypoints)} waypoints at Z={FLY_Z} m\n")

    # Live tracking canvas
    track_base = _make_canvas()
    # Draw planned path
    wp_px = [ned_to_canvas(n, e) for n, e in waypoints]
    if len(wp_px) > 1:
        pts = np.array(wp_px, dtype=np.int32)
        cv2.polylines(track_base, [pts], False, (0, 80, 200), 1)

    WP_REACH_DIST = 1.0   # metres — consider waypoint reached within this radius
    CMD_DT        = 0.15  # velocity command duration (s)

    async def goto(wn, we, label, reach=WP_REACH_DIST):
        """Fly to (wn, we) at FLY_Z, update display, return False if aborted."""
        while True:
            s   = drone.get_ground_truth_kinematics()
            p   = s["pose"]["position"]
            pn, pe, pz = float(p["x"]), float(p["y"]), float(p["z"])
            dn, de = wn - pn, we - pe
            dist   = math.hypot(dn, de)

            if dist < reach:
                return True

            scale = min(FLY_SPEED, dist) / dist
            vd    = float(np.clip((FLY_Z - pz) * 2.0, -FLY_SPEED, FLY_SPEED))
            await drone.move_by_velocity_async(
                v_north=dn * scale, v_east=de * scale,
                v_down=vd, duration=CMD_DT * 2
            )

            frame = track_base.copy()
            cv2.circle(frame, ned_to_canvas(pn, pe), 6, (0, 255, 0), -1)
            cv2.circle(frame, ned_to_canvas(wn, we), 10, (0, 255, 255), 2)
            cv2.putText(frame, f"{label}  dist={dist:.1f} m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2)
            cv2.putText(frame, "Esc to abort",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)
            cv2.imshow("Live Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False
            await asyncio.sleep(CMD_DT)

    try:
        # ── Phase 1: fly to trajectory start point ────────────────────────
        start_n, start_e = waypoints[0]
        print(f"  ↗ Flying to start point  N={start_n:.2f}  E={start_e:.2f}")
        if not await goto(start_n, start_e, "To start point", reach=WP_REACH_DIST):
            print("  Aborted.")
            return
        print("  ✓ At start point — following trajectory\n")

        # ── Phase 2: follow the trajectory ───────────────────────────────
        for i, (wn, we) in enumerate(waypoints):
            print(f"  → WP {i+1}/{len(waypoints)}  N={wn:.2f}  E={we:.2f}")

            # Velocity-based approach: steer toward waypoint until close enough
            while True:
                s   = drone.get_ground_truth_kinematics()
                p   = s["pose"]["position"]
                pn  = float(p["x"])
                pe  = float(p["y"])
                pz  = float(p["z"])

                dn   = wn - pn
                de   = we - pe
                dist = math.hypot(dn, de)

                if dist < WP_REACH_DIST:
                    break

                # Velocity toward waypoint (capped at FLY_SPEED)
                scale = min(FLY_SPEED, dist) / dist
                vn = dn * scale
                ve = de * scale
                # Gentle altitude correction
                vd = float(np.clip((FLY_Z - pz) * 2.0, -FLY_SPEED, FLY_SPEED))

                await drone.move_by_velocity_async(
                    v_north=vn, v_east=ve, v_down=vd, duration=CMD_DT * 2
                )

                # Update live display
                frame = track_base.copy()
                dpx   = ned_to_canvas(pn, pe)
                cv2.circle(frame, dpx, 6, (0, 255, 0), -1)
                cv2.circle(frame, ned_to_canvas(wn, we), 8, (0, 180, 255), 2)
                cv2.putText(frame, f"WP {i+1}/{len(waypoints)}  dist={dist:.1f} m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 0), 2)
                cv2.putText(frame, "Esc to abort",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (180, 180, 180), 1)
                cv2.imshow("Live Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("  Aborted by user.")
                    return

                await asyncio.sleep(CMD_DT)

        print("\nTrajectory complete!")

    finally:
        cv2.destroyAllWindows()
        print("Landing…")
        try:
            await (await drone.land_async())
        except Exception:
            pass
        drone.disarm()
        drone.disable_api_control()
        client.disconnect()
        print("Done.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    canvas_base = _make_canvas()
    cv2.namedWindow("Draw Trajectory")
    cv2.setMouseCallback("Draw Trajectory", _mouse_cb, canvas_base)
    cv2.imshow("Draw Trajectory", _redraw(canvas_base))

    print("Draw your trajectory on the canvas.")
    print("Left-drag to draw, Right-click to clear, Space/Enter to fly, Esc to quit.\n")

    while True:
        cv2.imshow("Draw Trajectory", _redraw(canvas_base))
        key = cv2.waitKey(30) & 0xFF
        if key == 27:   # Esc
            print("Quit.")
            break
        if key in (32, 13):   # Space or Enter
            wps = extract_waypoints()
            if len(wps) < 2:
                print("  Draw a longer path first (need at least 2 waypoints).")
                continue
            cv2.destroyAllWindows()
            asyncio.run(fly_trajectory(wps))
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
