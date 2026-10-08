import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import sys

# =========================================================
# Load MuJoCo model
# =========================================================

MODEL_FILE = "turtlebot_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(MODEL_FILE)
data = mujoco.MjData(model)

# =========================================================
# Robot speed
# =========================================================

SPEED = 8.0

left_speed = 0.0
right_speed = 0.0

# =========================================================
# Keyboard state
# =========================================================

keys_pressed = set()

# =========================================================
# Keyboard input for Windows
# =========================================================

if sys.platform == "win32":

    import msvcrt

    def keyboard_thread():

        global left_speed
        global right_speed

        while True:

            if msvcrt.kbhit():

                key = msvcrt.getch()

                # Arrow keys produce special codes
                if key == b'\xe0':

                    key2 = msvcrt.getch()

                    # UP
                    if key2 == b'H':
                        keys_pressed.add("up")

                    # DOWN
                    elif key2 == b'P':
                        keys_pressed.add("down")

                    # LEFT
                    elif key2 == b'K':
                        keys_pressed.add("left")

                    # RIGHT
                    elif key2 == b'M':
                        keys_pressed.add("right")

                elif key == b' ':

                    keys_pressed.clear()

            time.sleep(0.01)

else:

    # =====================================================
    # Linux / macOS keyboard input
    # =====================================================

    import curses

    def keyboard_thread():

        def main(stdscr):

            stdscr.nodelay(True)
            stdscr.keypad(True)

            while True:

                key = stdscr.getch()

                keys_pressed.clear()

                if key == curses.KEY_UP:
                    keys_pressed.add("up")

                elif key == curses.KEY_DOWN:
                    keys_pressed.add("down")

                elif key == curses.KEY_LEFT:
                    keys_pressed.add("left")

                elif key == curses.KEY_RIGHT:
                    keys_pressed.add("right")

                elif key == ord(' '):
                    keys_pressed.clear()

                time.sleep(0.01)

        curses.wrapper(main)


# Start keyboard thread
thread = threading.Thread(
    target=keyboard_thread,
    daemon=True
)

thread.start()


# =========================================================
# Convert keyboard input to wheel velocities
# =========================================================

def update_robot():

    global left_speed
    global right_speed

    left_speed = 0.0
    right_speed = 0.0

    # -----------------------------
    # Forward
    # -----------------------------

    if "up" in keys_pressed:

        left_speed = SPEED
        right_speed = SPEED

    # -----------------------------
    # Backward
    # -----------------------------

    elif "down" in keys_pressed:

        left_speed = -SPEED
        right_speed = -SPEED

    # -----------------------------
    # Turn left
    # -----------------------------

    elif "left" in keys_pressed:

        left_speed = -SPEED
        right_speed = SPEED

    # -----------------------------
    # Turn right
    # -----------------------------

    elif "right" in keys_pressed:

        left_speed = SPEED
        right_speed = -SPEED

    # Send commands to MuJoCo
    data.ctrl[0] = left_speed
    data.ctrl[1] = right_speed


# =========================================================
# Run MuJoCo
# =========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print()
    print("======================================")
    print(" TurtleBot Waffle Pi Controller")
    print("======================================")
    print()
    print("UP       -> Forward")
    print("DOWN     -> Backward")
    print("LEFT     -> Rotate Left")
    print("RIGHT    -> Rotate Right")
    print("SPACE    -> Stop")
    print()
    print("MuJoCo simulation started...")
    print()

    while viewer.is_running():

        # Update wheel commands
        update_robot()

        # Physics simulation
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        # Real-time simulation
        time.sleep(model.opt.timestep)
