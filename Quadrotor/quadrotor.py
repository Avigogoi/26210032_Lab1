import time
import numpy as np
import mujoco
import mujoco.viewer
import glfw


MODEL = "quadrotor.xml"

DT = 0.01

LINEAR_SPEED = 2.0
VERTICAL_SPEED = 1.5
YAW_SPEED = 1.5


model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)


# ---------------------------------------------------------
# Keyboard state
# ---------------------------------------------------------

keys = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False,
    "yaw_left": False,
    "yaw_right": False,
}


def key_callback(keycode):
    # Arrow keys
    if keycode == glfw.KEY_UP:
        keys["forward"] = True

    elif keycode == glfw.KEY_DOWN:
        keys["backward"] = True

    elif keycode == glfw.KEY_LEFT:
        keys["left"] = True

    elif keycode == glfw.KEY_RIGHT:
        keys["right"] = True

    elif keycode == glfw.KEY_SPACE:
        keys["up"] = True

    elif keycode == glfw.KEY_LEFT_SHIFT:
        keys["down"] = True

    elif keycode == glfw.KEY_Q:
        keys["yaw_left"] = True

    elif keycode == glfw.KEY_E:
        keys["yaw_right"] = True


# ---------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------

def rotation_matrix(yaw, pitch, roll):

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cr = np.cos(roll)
    sr = np.sin(roll)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])

    Rx = np.array([
        [1, 0,  0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    return Rz @ Ry @ Rx


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

x = 0.0
y = 0.0
z = 1.0

roll = 0.0
pitch = 0.0
yaw = 0.0


# ---------------------------------------------------------
# Main simulation
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback) as viewer:

    last_print = 0

    while viewer.is_running():

        # -------------------------------------------------
        # Keyboard velocities in BODY FRAME
        # -------------------------------------------------

        vx_body = 0.0
        vy_body = 0.0
        vz_world = 0.0
        yaw_rate = 0.0

        if keys["forward"]:
            vx_body += LINEAR_SPEED

        if keys["backward"]:
            vx_body -= LINEAR_SPEED

        if keys["left"]:
            vy_body += LINEAR_SPEED

        if keys["right"]:
            vy_body -= LINEAR_SPEED

        if keys["up"]:
            vz_world += VERTICAL_SPEED

        if keys["down"]:
            vz_world -= VERTICAL_SPEED

        if keys["yaw_left"]:
            yaw_rate += YAW_SPEED

        if keys["yaw_right"]:
            yaw_rate -= YAW_SPEED


        # -------------------------------------------------
        # Body -> World transformation
        # -------------------------------------------------

        R_WB = rotation_matrix(
            yaw,
            pitch,
            roll
        )

        velocity_body = np.array([
            vx_body,
            vy_body,
            0.0
        ])

        velocity_world = R_WB @ velocity_body


        # -------------------------------------------------
        # Update position
        # -------------------------------------------------

        x += velocity_world[0] * DT
        y += velocity_world[1] * DT
        z += vz_world * DT

        z = max(0.2, z)


        # -------------------------------------------------
        # Yaw
        # -------------------------------------------------

        yaw += yaw_rate * DT


        # -------------------------------------------------
        # Small visual attitude change
        #
        # This makes body-frame changes easier to see.
        # -------------------------------------------------

        target_pitch = -vx_body * 0.08
        target_roll = vy_body * 0.08

        pitch += (target_pitch - pitch) * 0.08
        roll += (target_roll - roll) * 0.08


        # -------------------------------------------------
        # Convert rotation matrix to MuJoCo quaternion
        # -------------------------------------------------

        R_WB = rotation_matrix(
            yaw,
            pitch,
            roll
        )

        quat = np.zeros(4)

        mujoco.mju_mat2Quat(
            quat,
            R_WB.reshape(-1)
        )


        # -------------------------------------------------
        # Write pose into MuJoCo
        # -------------------------------------------------

        data.qpos[0] = x
        data.qpos[1] = y
        data.qpos[2] = z

        data.qpos[3:7] = quat

        mujoco.mj_forward(model, data)


        # -------------------------------------------------
        # Display information
        # -------------------------------------------------

        if time.time() - last_print > 0.1:

            print("\033[2J\033[H", end="")

            print("QUADROTOR — MuJoCo")
            print("==================")

            print()
            print("Keyboard:")
            print("  ↑       Forward")
            print("  ↓       Backward")
            print("  ←       Left")
            print("  →       Right")
            print("  SPACE   Up")
            print("  SHIFT   Down")
            print("  Q       Yaw left")
            print("  E       Yaw right")

            print()
            print("World position:")
            print(
                f"  x = {x: .3f}"
                f"  y = {y: .3f}"
                f"  z = {z: .3f}"
            )

            print()
            print("Attitude:")
            print(
                f"  Roll  = {np.degrees(roll): .2f}°"
            )
            print(
                f"  Pitch = {np.degrees(pitch): .2f}°"
            )
            print(
                f"  Yaw   = {np.degrees(yaw): .2f}°"
            )

            print()
            print("R_WB  (World <- Body)")
            print("--------------------------------")

            for row in R_WB:
                print(
                    "  "
                    + " ".join(
                        f"{v: .4f}" for v in row
                    )
                )

            print()
            print("Body X axis in WORLD:")
            print(
                " ",
                np.round(R_WB[:, 0], 4)
            )

            print("Body Y axis in WORLD:")
            print(
                " ",
                np.round(R_WB[:, 1], 4)
            )

            print("Body Z axis in WORLD:")
            print(
                " ",
                np.round(R_WB[:, 2], 4)
            )

            print()
            print("Body velocity:")
            print(
                f"  [{vx_body:.2f}, "
                f"{vy_body:.2f}, 0.00]"
            )

            print()
            print("World velocity:")
            print(
                f"  [{velocity_world[0]:.2f}, "
                f"{velocity_world[1]:.2f}, "
                f"{velocity_world[2]:.2f}]"
            )

            last_print = time.time()


        viewer.sync()

        time.sleep(DT)
