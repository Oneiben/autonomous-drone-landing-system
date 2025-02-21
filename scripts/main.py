import cv2
import numpy as np
from control_actions import Control
from image_processing import image_processing
from simulation import UnityEnvironmentWrapper


class Main:
    def __init__(self, unity_env_path):
        """
        Initialize the main control class with Unity environment and control logic.

        Args:
            unity_env_path (str): Path to the Unity environment application.
        """
        # Initialize Unity environment, control system, and image processing
        self.unity_env = UnityEnvironmentWrapper(unity_env_path)
        self.control = Control(kp_min=0.015, kp_max=0.015, error_threshold=15)
        self.image_processing = image_processing()

        # Flag to check if the simulation has ended
        self.done = False

        # Flag to indicate if the emergency landing is in progress
        self.landing_in_progress = False

    def run(self):
        """
        Main loop for running the simulation and controlling the actions.
        Continuously process the camera image, calculate control actions,
        and interact with the Unity environment.
        """
        observation = self.unity_env.reset()

        try:
            while not self.done:
                # Get the camera image from the Unity environment
                camera_image = np.transpose(
                    observation[0], (1, 2, 0)
                )  # Transpose to (height, width, channels)
                frame = cv2.cvtColor(
                    camera_image, cv2.COLOR_RGB2BGR
                )  # Convert RGB to BGR for OpenCV

                # Extract position sensor data (e.g., height)
                Position_Sensor = observation[1]
                height = Position_Sensor[1]

                # Check for emergency landing mode (activated when '0' key is pressed)
                if cv2.waitKey(1) & 0xFF == ord("0"):
                    self.landing_in_progress = True
                    print("Emergency landing mode activated!")

                if self.landing_in_progress:
                    # If emergency landing mode is on, control the throttle to land
                    throttle_land = self.control.throttle_control(height)
                    actions = [
                        0.0,
                        0.0,
                        0.0,
                        -throttle_land,
                    ]  # Set all other control values to 0, except throttle

                else:
                    # Normal operation: Get control actions based on the current image frame and height
                    actions, annotated_frame = self.control.get_control_actions(
                        frame, height
                    )

                # Step the environment with the calculated actions
                observation, reward, done, info = self.unity_env.step(actions)

                # Check if the environment is done (simulation ended)
                if done:
                    self.done = True

                # Display annotated frame (for debugging and visual monitoring)
                cv2.imshow("Downward Camera View", annotated_frame)
                cv2.moveWindow("Downward Camera View", 500, 250)

                # Reset environment if done
                if self.done:
                    observation = self.unity_env.reset()

                # Exit on ESC key press
                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key ASCII code
                    break

        finally:
            # Close Unity environment and OpenCV windows properly at the end
            self.unity_env.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the Main class and run the simulation

    app = Main("./simulations/linux_build/Linux_Simulation.x86_64")  # for Linux
    # app = Main("./simulations/macos_build/MacOS_Simulation.app")  # for MacOS
    # app = Main("./simulations/windows_build/Xerox_UAV.exe")         # for Windows

    app.run()
