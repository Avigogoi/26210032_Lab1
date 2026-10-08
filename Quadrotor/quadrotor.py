import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# LOAD XML
# ============================================================

model = mujoco.MjModel.from_xml_path("quadrotor.xml")
data = mujoco.MjData(model)


# ============================================================
# QUADROTOR BODY ID
# ============================================================

quad_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "quadrotor"
)

print("Quadrotor body ID =", quad_id)


# ============================================================
# VELOCITY
# ============================================================

velocity = np.array([0.0, 0.0, 0.0])

speed = 1.0


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def key_callback(keycode):

    global velocity

    # UP ARROW
    if keycode == 265:
        velocity[0] = speed
        print("FORWARD")

    # DOWN ARROW
    elif keycode == 264:
        velocity[0] = -speed
        print("BACKWARD")

    # LEFT ARROW
    elif keycode == 263:
        velocity[1] = speed
        print("LEFT")

    # RIGHT ARROW
    elif keycode == 262:
        velocity[1] = -speed
        print("RIGHT")

    # SPACE
    elif keycode == 32:
        velocity[2] = speed
        print("UP")

    # SHIFT
    elif keycode == 340:
        velocity[2] = -speed
        print("DOWN")

    # R KEY
    elif keycode == 82 or keycode == 114:

        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 1.0

        data.qvel[:] = 0.0

        velocity[:] = 0.0

        print("RESET")


# ============================================================
# START VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback) as viewer:

    print()
    print("======================================")
    print("       QUADROTOR CONTROL")
    print("======================================")
    print("UP       = Forward")
    print("DOWN     = Backward")
    print("LEFT     = Left")
    print("RIGHT    = Right")
    print("SPACE    = Up")
    print("SHIFT    = Down")
    print("R        = Reset")
    print("======================================")
    print()

    while viewer.is_running():

        # ----------------------------------------------------
        # SET VELOCITY
        # ----------------------------------------------------

        data.qvel[0] = velocity[0]
        data.qvel[1] = velocity[1]
        data.qvel[2] = velocity[2]


        # ----------------------------------------------------
        # CANCEL ROTATION
        # ----------------------------------------------------

        data.qvel[3] = 0.0
        data.qvel[4] = 0.0
        data.qvel[5] = 0.0


        # ----------------------------------------------------
        # STEP SIMULATION
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)
