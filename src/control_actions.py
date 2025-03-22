import numpy as np
from image_processing import image_processing

class Control:
    def __init__(self, kp_min=0.015, kp_max=0.015, error_threshold=15):
        """
        Initialize the Control class with proportional control parameters.

        Args:
            kp_min (float): Minimum proportional gain for roll and pitch control.
            kp_max (float): Maximum proportional gain for roll and pitch control.
            error_threshold (int): Threshold for error, below which throttle is decreased.
        """
        self.kp_min = kp_min
        self.kp_max = kp_max
        self.image_processing = image_processing()
        self.error_threshold = error_threshold
        self.exploration_throttle = 0.5  # Throttle value during exploration phase.
        self.landing_throttle = 0.81  # Throttle value for landing phase.
        self.landing_mode = False
        self.actions = [0.0, 0.0, 0.0, 0.4]  # Initial control actions [roll, pitch, yaw, throttle].
        
        # Stores the last detected rectangle center for continuity in tracking.
        self.last_rectangle_center = None

    def compute_error(self, image_center, rectangle_center):
        """
        Compute the positional error between the image center and the detected rectangle center.

        Args:
            image_center (list): [x, y] coordinates of the image center.
            rectangle_center (list): [x, y] coordinates of the detected rectangle center.

        Returns:
            tuple: (error_x, error_y) representing errors in the x-axis and y-axis.
        """
        error_x = rectangle_center[0] - image_center[0]
        error_y = rectangle_center[1] - image_center[1]
        return error_x, error_y

    def adjust_gains(self, error_x, error_y):
        """
        Dynamically adjust the proportional gains based on error magnitude.

        Note:
            Currently, this function does not modify `kp_roll` and `kp_pitch`, 
            as they are set to fixed values (`kp_min` and `kp_max`). However, 
            if gain adjustment is needed, this function can be modified to 
            dynamically update `kp_roll` and `kp_pitch` based on error magnitude.
            
            To enable dynamic gain adjustment:
            - Modify `self.kp_roll` and `self.kp_pitch` within this function.
            - Update `kp_min` and `kp_max` accordingly to define the range of variation.

        Args:
            error_x (float): Error in the x-axis.
            error_y (float): Error in the y-axis.

        Returns:
            tuple: (kp_roll, kp_pitch) proportional gains for roll and pitch.
        """
        self.kp_roll = self.kp_min  # Currently fixed, but can be adjusted dynamically.
        self.kp_pitch = self.kp_max  # Currently fixed, but can be adjusted dynamically.

        # Scaling factors for gain adjustment (currently unused but can be leveraged)
        scale_factor_x = -error_x / 80  
        scale_factor_y = -error_y / 80  

        # If dynamic gain control is needed, uncomment and modify the following lines:
        # self.kp_roll = max(self.kp_min, min(self.kp_max, self.kp_roll + scale_factor_x))
        # self.kp_pitch = max(self.kp_min, min(self.kp_max, self.kp_pitch + scale_factor_y))

        return self.kp_roll, self.kp_pitch

    def decrease_error(self, error_x, error_y):
        """
        Convert positional error into control actions.

        Args:
            error_x (float): Error in the x-axis (roll).
            error_y (float): Error in the y-axis (pitch).

        Returns:
            list: Control actions [roll, pitch, yaw, throttle].
        """
        # Adjust gains based on error magnitude.
        adjusted_kp_roll, adjusted_kp_pitch = self.adjust_gains(error_x, error_y)

        # Compute control actions.
        roll = adjusted_kp_roll * error_x / 10
        pitch = -adjusted_kp_pitch * error_y / 10
        yaw = 0.0  # No yaw adjustment for now.

        return [roll, pitch, yaw, 0]

    def throttle_control(self, pos):
        """
        Adjust throttle based on altitude or position.

        Args:
            pos (float): Current altitude or position.

        Returns:
            float: Adjusted throttle value.
        """
        ground = 0.1  # Define ground level threshold.
        if pos <= ground:
            return 0.0  # If near ground, set throttle to zero.
        else:
            # Reduce throttle smoothly to avoid sudden drops.
            self.landing_throttle = max(0.05, pos / 1.5)
            return self.landing_throttle

    def get_control_actions(self, frame, pos):
        """
        Process an image frame to calculate control actions.

        Args:
            frame (numpy.ndarray): Camera frame used for image processing.
            pos (float): Current altitude or position.

        Returns:
            tuple: (control actions, annotated frame with visual indicators).
        """
        image_center, rectangle_center, annotated_frame = self.image_processing.cv_detection(frame)
        # Alternative detection methods (uncomment if needed):
        # image_center, rectangle_center, annotated_frame = self.image_processing.tes_detection(frame)
        # image_center, rectangle_center, annotated_frame = self.image_processing.Yolo_detection(frame)

        # Adjust throttle based on altitude.
        throttle_land = self.throttle_control(pos)

        if rectangle_center is not None:
            self.landing_mode = True
            # Store the last detected rectangle center.
            self.last_rectangle_center = rectangle_center
        elif self.last_rectangle_center is not None:
            # Use the last known position if detection is lost.
            rectangle_center = self.last_rectangle_center
            self.landing_mode = True

        if self.landing_mode:
            if rectangle_center is not None:
                # Compute error values for roll and pitch.
                error_x, error_y = self.compute_error(image_center, rectangle_center)
                if abs(error_x) > self.error_threshold and abs(error_y) > self.error_threshold:
                    self.actions = self.decrease_error(error_x, error_y)
                else:
                    # If error is below threshold, reduce throttle.
                    self.actions = [0.0, 0.0, 0.0, -throttle_land]
            else:
                # If detection is lost, maintain last throttle control.
                self.actions = [0.0, 0.0, 0.0, -throttle_land]
        else:
            # Maintain exploration throttle when not in landing mode.
            self.actions = [0.0, 0.0, 0.0, self.exploration_throttle]

        return self.actions, annotated_frame
