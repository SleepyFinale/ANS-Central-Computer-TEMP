import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import time

# TF imports
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')
        self.get_logger().info("Explorer Node Started")

        # Parameters (allow selecting the cartographer map topic and frames)
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_queue_size', 10)

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        map_queue = self.get_parameter('map_queue_size').get_parameter_value().integer_value

        # Subscriber to the map topic (configurable)
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, map_queue)

        # Action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Visited frontiers set
        self.visited_frontiers = set()

        # Track recently failed frontiers to avoid retrying immediately
        self.failed_frontiers = {}  # centroid -> last_failed_time (seconds)
        
        # Track recently explored areas to avoid revisiting same regions
        self.recently_explored_areas = []  # List of (row, col, timestamp) for recently explored positions
        self.explored_area_retention_time = 60.0  # seconds; how long to remember explored areas
        self.explored_area_penalty_radius = 30  # cells; penalty for frontiers within this radius of recently explored areas
        
        # Track if robot has been in same area too long
        self.last_area_change_time = time.time()
        self.last_area_position = None
        self.area_stagnation_timeout = 20.0  # seconds; if in same area for this long, force exploration elsewhere
        self.area_stagnation_radius = 20  # cells; consider same area if within this radius

        # Minimum frontier cluster size to prefer (cells)
        self.min_frontier_size = 0

        # Current goal centroid (r, c) in map cells
        self.current_goal = None

        # Map and position data
        self.map_data = None

        # TF buffer and listener to obtain robot pose in the map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # TF broadcaster to publish temporary odom corrections during checkpoint validation
        self.tf_broadcaster = TransformBroadcaster(self)

        # robot_position stored in map cell coordinates (row, col)
        self.robot_position = (0, 0)
        
        # Track idle time to penalize standing still
        self.last_position = (0, 0)
        self.last_position_time = time.time()
        self.idle_threshold_distance_cells = 10  # cells; consider idle if moved less than this (approx 0.5m at 0.05m resolution)
        self.idle_threshold_time = 8.0  # seconds; consider idle after standing for this long (reduced for faster movement)
        
        # NEW: Maximum pause timeout to prevent waiting for Cartographer optimizations
        # Forces movement even if SLAM confidence is not perfect (safety valve for speed)
        self.last_goal_sent_time = time.time()
        self.max_goal_pause_time = 1.0  # seconds; force new frontier search if no movement in 1s (reduced for faster movement)
        self.consecutive_paused_iterations = 0  # Track how many times we've been paused
        
        # Movement detection: track if robot is actually moving toward goal
        self.last_movement_check_time = time.time()
        self.movement_check_interval = 1.0  # Check movement every 1 second (reduced for faster detection)
        self.last_movement_position = None  # Last position when we checked movement
        self.stuck_movement_threshold = 0.1  # meters; if moved less than this, consider stuck
        self.stuck_timeout = 2.0  # seconds; if stuck for this long, force new goal (reduced for faster recovery)
        
        # SENSOR FUSION: Enhanced odometry and IMU integration
        # Track odometry confidence and drift estimation over time
        self.odom_drift_estimate = 0.0  # Estimated cumulative drift (meters)
        self.last_odom_pose = None  # Previous odometry pose for drift tracking
        self.imu_yaw_offset = 0.0  # IMU yaw bias correction
        self.imu_samples = []  # Ring buffer of recent IMU measurements
        self.max_imu_samples = 50  # Keep last 50 IMU samples for drift detection
        self.odom_weight = 0.7  # Trust level for odometry (0.0-1.0), will adjust based on confidence
        self.imu_weight = 0.3  # Trust level for IMU (0.0-1.0)
        self.loop_closure_detected = False  # Flag: loop closure just occurred
        self.last_loop_closure_time = 0.0  # Time of last confirmed loop closure
        
        # FEATURE TRACKING: For loop closure and sensor fusion validation
        self.observed_features = {}  # {feature_id: [(x,y,timestamp), ...]} - track feature observations
        self.feature_counter = 0  # Unique ID for new features
        self.max_features_tracked = 100  # Limit memory usage
        # WALL PREFERENCE: Prefer frontiers near walls/outlines before open areas
        self.wall_preference = 0.8  # 0.0=no preference, 1.0=strong preference for walls
        self.wall_proximity_distance = 3  # cells within which a frontier cell is considered 'near a wall'
        
        # DEPRECATED: Breadcrumb and checkpoint logic (disabled for sensor fusion focus)
        # self.breadcrumb_waypoints = []
        # self.exploration_distance = 0.0
        # self.last_breadcrumb_distance = 0.0
        # self.breadcrumb_interval = 50.0
        # self.force_return_distance = 200.0
        # self.returning_to_checkpoint = False
        # self.checkpoint_validation_time = None
        # self.checkpoint_validation_timeout = 30.0
        # self.checkpoint_attempts = 0
        # self.checkpoint_max_attempts = 3

        # Timer for periodic exploration (faster decision-making)
        self.timer = self.create_timer(0.3, self.explore)  # Reduced from 0.5s for faster decision-making

        # Cache last TF update time to avoid excessive lookups
        self.last_tf_update = 0.0
        self.tf_update_interval = 0.2  # Update robot position every 200ms
        
        # Map stability monitoring: detect sudden map corrections/jumps
        self.last_map_odom_transform = None
        self.map_jump_threshold = 0.5  # meters; if map->odom changes by more than this, it's a jump
        self.map_jump_detected = False
        self.map_jump_time = 0.0
        self.map_stabilization_timeout = 1.0  # seconds to wait after map jump before resuming (reduced for faster movement)
        
        # CRITICAL: Drift monitoring - detect when odom drifts away from map
        # This indicates the robot has lost localization (common around curved objects)
        # The map->odom transform represents how much odom has drifted from the map frame
        # We want to keep this as close to zero as possible
        self.map_odom_drift_threshold = 0.0  # meters; if drift exceeds this, something is wrong (was 0.3)
        self.map_odom_drift_detected = False
        self.drift_recovery_mode = False
        self.last_map_odom_position = None
        self.drift_check_interval = 1.0  # Check drift every second
        self.last_drift_check = 0.0
        
        # Active drift correction: navigate to help Cartographer recalibrate
        self.drift_correction_goal = None  # Goal position for drift correction
        self.drift_correction_active = False  # Flag: actively correcting drift
        self.drift_correction_timeout = 15.0  # Max time to spend correcting drift (reduced for faster movement)
        self.drift_correction_start_time = 0.0
        self.drift_correction_threshold = 0.15  # meters; drift below this is acceptable
        
        # Post-recovery behavior: prevent getting stuck trying to refine localization
        self.drift_recovery_time = 0.0  # When drift was last recovered
        self.post_recovery_movement_threshold = 0.5  # meters; must move this far after recovery
        self.post_recovery_timeout = 5.0  # seconds; if stuck refining after this, move on (reduced for faster movement)
        self.post_recovery_position = None  # Position when recovery completed
        self.in_post_recovery_mode = False  # Flag: recently recovered from drift
        
        # Open space detection: track when robot is in featureless areas
        self.last_scan_time = time.time()
        self.scan_timeout = 1.0  # If no valid scans for this long, we're likely in open space
        self.open_space_detected = False
        
        # Dynamic feature detection: monitor LiDAR feature density
        self.feature_density_threshold = 0.05  # Minimum feature density to trust LiDAR (lowered from 0.1 for more lenient detection)
        self.low_feature_area_detected = False
        self.feature_density_history = []  # Track recent feature densities
        self.max_feature_history = 10  # Keep last 10 measurements
        self.low_feature_consecutive_count = 0  # Count consecutive low-feature detections
        self.low_feature_required_count = 3  # Require this many consecutive detections before flagging

    def map_callback(self, msg):
        self.map_data = msg
        # Log map receipt so we can see incoming updates in the main logs
        # self.get_logger().info("Map received")

    def monitor_map_stability(self):
        """
        Monitor map->odom transform for sudden jumps that indicate loop closure corrections.
        If a large jump is detected, pause exploration briefly to let the map stabilize.
        Also monitors for drift - if odom drifts away from map, robot has lost localization.
        """
        map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        odom_frame = 'odom'
        robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        
        try:
            now = rclpy.time.Time()
            trans_map_odom: TransformStamped = self.tf_buffer.lookup_transform(
                map_frame, odom_frame, now, timeout=Duration(seconds=0.5))
            
            current_x = trans_map_odom.transform.translation.x
            current_y = trans_map_odom.transform.translation.y
            
            # Check for map jumps (loop closure corrections)
            if self.last_map_odom_transform is not None:
                dx = current_x - self.last_map_odom_transform[0]
                dy = current_y - self.last_map_odom_transform[1]
                jump_distance = np.sqrt(dx**2 + dy**2)
                
                if jump_distance > self.map_jump_threshold:
                    self.map_jump_detected = True
                    self.map_jump_time = time.time()
                    self.get_logger().warning(
                        f"Map jump detected! Distance: {jump_distance:.2f}m "
                        f"(dx={dx:.2f}, dy={dy:.2f}). Pausing exploration to stabilize."
                    )
            
            self.last_map_odom_transform = (current_x, current_y)
            
            # CRITICAL: Monitor drift - if map->odom transform is large, robot has lost localization
            # The map->odom transform should be small if robot is well-localized
            # Large values indicate odom has drifted away from the map
            # We want to actively correct this by navigating to help Cartographer recalibrate
            drift_magnitude = np.sqrt(current_x**2 + current_y**2)
            
            now_time = time.time()
            if now_time - self.last_drift_check > self.drift_check_interval:
                self.last_drift_check = now_time
                
                if drift_magnitude > self.map_odom_drift_threshold:
                    if not self.map_odom_drift_detected:
                        self.map_odom_drift_detected = True
                        self.drift_recovery_mode = True
                        self.drift_correction_active = True
                        self.drift_correction_start_time = now_time
                        # Clear any existing goal
                        if self.current_goal is not None:
                            self.current_goal = None
                        self.get_logger().error(
                            f"CRITICAL: Map->odom drift detected! Magnitude: {drift_magnitude:.2f}m "
                            f"(x={current_x:.2f}, y={current_y:.2f}). "
                            f"Robot has likely lost localization. Starting active drift correction."
                        )
                        # Calculate correction goal: navigate to a nearby location with good features
                        # This helps Cartographer recalibrate by providing known reference points
                        # Simple implementation: use robot's current position as reference
                        if self.robot_position is not None and self.map_data is not None:
                            robot_row, robot_col = self.robot_position
                            goal_x = robot_col * self.map_data.info.resolution + self.map_data.info.origin.position.x
                            goal_y = robot_row * self.map_data.info.resolution + self.map_data.info.origin.position.y
                            self.drift_correction_goal = (goal_x, goal_y)
                        else:
                            self.drift_correction_goal = None
                elif drift_magnitude > self.drift_correction_threshold:
                    # Drift is still present but below critical threshold
                    # Continue active correction if we're in correction mode
                    if self.drift_correction_active and self.drift_correction_goal is None:
                        # Recalculate goal if we don't have one - use current position
                        if self.robot_position is not None and self.map_data is not None:
                            robot_row, robot_col = self.robot_position
                            goal_x = robot_col * self.map_data.info.resolution + self.map_data.info.origin.position.x
                            goal_y = robot_row * self.map_data.info.resolution + self.map_data.info.origin.position.y
                            self.drift_correction_goal = (goal_x, goal_y)
                else:
                    # Drift is below acceptable threshold
                    if self.map_odom_drift_detected or self.drift_correction_active:
                        self.map_odom_drift_detected = False
                        self.drift_recovery_mode = False
                        self.drift_correction_active = False
                        self.drift_correction_goal = None
                        self.drift_recovery_time = time.time()
                        self.in_post_recovery_mode = True
                        # Record position when recovery completed
                        if self.robot_position is not None:
                            self.post_recovery_position = self.robot_position
                        self.get_logger().warning(
                            f"Map->odom drift corrected! Final magnitude: {drift_magnitude:.2f}m. "
                            f"Exiting recovery mode. Will move on if stuck refining localization."
                        )
            
            # Store current position for drift tracking
            self.last_map_odom_position = (current_x, current_y)
            
        except Exception as e:
            # TF lookup failed, skip monitoring this cycle
            self.get_logger().warning(f"Map stability monitoring failed: {e}")
    
    def is_map_stable(self):
        """
        Check if map has stabilized after a detected jump.
        Returns True if map is stable and exploration can proceed.
        """
        if not self.map_jump_detected:
            return True
        
        elapsed = time.time() - self.map_jump_time
        if elapsed > self.map_stabilization_timeout:
            self.map_jump_detected = False
            self.get_logger().warning("Map has stabilized. Resuming exploration.")
            return True
        
        return False
    
    def detect_low_feature_area(self, map_array):
        """
        Detect if robot is in a low-feature area (like near rounded objects).
        Returns True if in low-feature area, False otherwise.
        Uses feature density around robot position.
        """
        if self.robot_position is None or map_array is None:
            return False
        
        robot_row, robot_col = self.robot_position
        rows, cols = map_array.shape
        
        # Check a local area around robot (larger radius for better averaging)
        check_radius = 7  # cells (increased from 5 for more lenient detection, ~0.35m radius at 0.05m resolution)
        feature_count = 0
        total_cells = 0
        
        for dr in range(-check_radius, check_radius + 1):
            for dc in range(-check_radius, check_radius + 1):
                r = robot_row + dr
                c = robot_col + dc
                
                if 0 <= r < rows and 0 <= c < cols:
                    total_cells += 1
                    cell_val = int(map_array[r, c])
                    # Count occupied cells (walls/obstacles) as features
                    if cell_val >= 50:
                        feature_count += 1
        
        if total_cells == 0:
            return False
        
        feature_density = feature_count / total_cells
        
        # Track feature density history
        self.feature_density_history.append(feature_density)
        if len(self.feature_density_history) > self.max_feature_history:
            self.feature_density_history.pop(0)
        
        # Use average of recent measurements to avoid false positives
        avg_density = np.mean(self.feature_density_history) if self.feature_density_history else feature_density
        
        # Low feature area if density is below threshold
        # Require multiple consecutive detections before flagging to avoid false positives
        is_currently_low = avg_density < self.feature_density_threshold
        
        if is_currently_low:
            self.low_feature_consecutive_count += 1
        else:
            self.low_feature_consecutive_count = 0
        
        # Only flag as low-feature if we've seen it consistently
        is_low_feature = self.low_feature_consecutive_count >= self.low_feature_required_count
        
        if is_low_feature and not self.low_feature_area_detected:
            self.low_feature_area_detected = True
            self.get_logger().warning(
                f"Low feature area detected! Feature density: {avg_density:.3f} "
                f"(threshold: {self.feature_density_threshold}, consecutive: {self.low_feature_consecutive_count}). "
                f"Relying more on IMU/odometry."
            )
        elif not is_low_feature and self.low_feature_area_detected:
            self.low_feature_area_detected = False
            self.low_feature_consecutive_count = 0  # Reset counter when exiting
            self.get_logger().warning("Exited low feature area. Resuming normal operation.")
        
        return is_low_feature

    def update_robot_position_from_tf(self):
        """
        Lookup transform from map frame to robot frame and update
        `self.robot_position` in map cell coordinates (row, col).
        Uses throttling to avoid excessive TF lookups.
        """
        if self.map_data is None:
            return

        now_time = time.time()
        if now_time - self.last_tf_update < self.tf_update_interval:
            return  # Skip if updated recently

        self.last_tf_update = now_time

        map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value

        try:
            # Use timeout to avoid blocking if TF isn't ready
            now = rclpy.time.Time()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                map_frame, robot_frame, now, timeout=Duration(seconds=0.5))
        except Exception as e:
            # Could be missing transform; just skip updating position
            self.get_logger().debug(f"TF lookup failed: {e}")
            return

        x = trans.transform.translation.x
        y = trans.transform.translation.y

        origin = self.map_data.info.origin.position
        res = self.map_data.info.resolution

        col = int(np.floor((x - origin.x) / res))
        row = int(np.floor((y - origin.y) / res))

        # clamp into map bounds
        row = max(0, min(self.map_data.info.height - 1, row))
        col = max(0, min(self.map_data.info.width - 1, col))

        self.robot_position = (row, col)

    def navigate_to(self, x, y):
        """
        Send navigation goal to Nav2.
        """
        # Check if server is available with timeout to avoid blocking
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning("Nav2 action server not available, skipping goal")
            return
        
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0  # Facing forward

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        self.get_logger().debug(f"Navigating to goal: x={x}, y={y}")

        # Send the goal and register a callback for the result
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the goal response and attach a callback to the result.
        """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            # mark current goal as failed so we won't immediately retry
            if self.current_goal is not None and self.current_goal != "drift_correction":
                self.failed_frontiers[self.current_goal] = time.time()
            self.current_goal = None
            # If drift correction goal was rejected, clear it and let next cycle recalculate
            if self.drift_correction_active:
                self.drift_correction_goal = None
            return

        self.get_logger().debug("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """
        Callback to handle the result of the navigation action.
        """
        try:
            result = future.result().result
            self.get_logger().debug(f"Navigation completed with result: {result}")
            # On (apparent) failure we may want to mark it; many nav2 results
            # include a status or error code, but to be conservative mark as
            # completed only if result exists. If navigation failed an
            # exception will be raised and handled below.
            # If this was a drift correction goal, check if drift is resolved
            if self.current_goal == "drift_correction":
                self.get_logger().debug("Drift correction navigation completed. Checking if drift is resolved...")
                # The drift monitoring will check and clear this flag if drift is resolved
            self.current_goal = None
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")
            # mark current goal as failed so we don't keep retrying
            if self.current_goal is not None and self.current_goal != "drift_correction":
                self.failed_frontiers[self.current_goal] = time.time()
            self.current_goal = None
            # If drift correction goal failed, clear it and let next cycle recalculate
            if self.drift_correction_active:
                self.drift_correction_goal = None

    def find_frontiers(self, map_array):
        """
        Detect frontiers in the occupancy grid map and cluster them.
        Includes edges of the map as potential frontiers.

        Returns a list of clusters where each cluster is a dict:
        { 'cells': [(r,c), ...], 'centroid': (r_avg, c_avg), 'size': n }
        """
        rows, cols = map_array.shape

        # raw frontier cells
        # Note: OccupancyGrid values: -1 = unknown, 0 = free, 100 = occupied
        # Standard frontier definition: a free cell adjacent to at least one unknown cell
        frontier_cells = set()

        for r in range(rows):
            for c in range(cols):
                val = int(map_array[r, c])
                # Only consider free cells as candidate frontier cells
                if val != 0:
                    continue

                is_frontier = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        # If neighbor is outside the map -> treat as frontier (edge)
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            is_frontier = True
                            break
                        neighbor_val = int(map_array[nr, nc])
                        # If neighbor is unknown, this free cell is a frontier
                        if neighbor_val == -1:
                            is_frontier = True
                            break
                    if is_frontier:
                        break

                if is_frontier:
                    frontier_cells.add((r, c))

        # cluster frontier cells using simple BFS on 8-neighborhood
        clusters = []
        visited = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            stack = [cell]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                r0, c0 = cur
                # explore neighbors
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r0 + dr, c0 + dc)
                        if nb in frontier_cells and nb not in visited:
                            stack.append(nb)

            if comp:
                comp_arr = np.array(comp)
                centroid_r = int(np.round(comp_arr[:, 0].mean()))
                centroid_c = int(np.round(comp_arr[:, 1].mean()))
                clusters.append({'cells': comp, 'centroid': (centroid_r, centroid_c), 'size': len(comp)})

        # Always log the total clusters and cells found (helps debugging when zero)
        # self.get_logger().info(f"Found {len(clusters)} frontier clusters (total cells: {len(frontier_cells)})")
        return clusters

    def is_frontier_reachable(self, frontier_cells, map_array):
        """
        Simple reachability check: verify that at least one frontier cell
        has a path of free/unknown cells between it and the robot.
        Uses BFS to detect if frontier is behind an obstacle.
        Returns True if likely reachable, False if likely blocked.
        """
        if not frontier_cells or self.robot_position is None:
            return True  # Assume reachable if can't check

        robot_row, robot_col = self.robot_position
        rows, cols = map_array.shape

        # BFS from robot position to check reachability to any frontier cell
        visited = set()
        queue = [(robot_row, robot_col)]
        visited.add((robot_row, robot_col))
        frontier_set = set(frontier_cells)

        while queue:
            r, c = queue.pop(0)
            
            # If we reached any frontier cell, it's reachable
            if (r, c) in frontier_set:
                return True
            
            # Explore 8-connected neighbors
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in visited:
                        continue
                    
                    visited.add((nr, nc))
                    cell_val = int(map_array[nr, nc])
                    # Only traverse free (0) cells. Nav2 is configured with
                    # `allow_unknown: false` so planning through unknown cells
                    # will be rejected by the planner. Restrict reachability
                    # to known free space to avoid proposing unreachable goals.
                    if cell_val == 0:
                        queue.append((nr, nc))
        
        return False  # No path found; frontier likely blocked

    def get_cluster_safety_score(self, frontier_cells, map_array):
        """
        Compute safety score: penalize frontiers too close to walls.
        Returns a multiplier (0.0-1.0) where 1.0 = safe, 0.0 = too close to obstacles.
        """
        if not frontier_cells:
            return 1.0
        
        safe_distance = 2  # cells; require at least this distance from obstacles
        rows, cols = map_array.shape
        
        unsafe_count = 0
        for r, c in frontier_cells:
            # Check if any occupied cells are within safe_distance
            for dr in range(-safe_distance, safe_distance + 1):
                for dc in range(-safe_distance, safe_distance + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        cell_val = int(map_array[nr, nc])
                        if cell_val >= 50:  # occupied or likely occupied
                            unsafe_count += 1
                            break
        
        # If most cells are too close to walls, penalize heavily
        safety_ratio = max(0.0, 1.0 - (unsafe_count / max(1, len(frontier_cells))))
        return safety_ratio ** 2  # quadratic penalty for unsafe frontiers

    def get_unknown_space_density(self, frontier_cells, map_array):
        """
        Calculate the density of unknown space around a frontier cluster.
        Returns a ratio (0.0-1.0) indicating how much unknown space is nearby.
        Higher values mean more potential for map expansion.
        """
        if not frontier_cells:
            return 0.0
        
        rows, cols = map_array.shape
        unknown_count = 0
        total_checked = 0
        check_radius = 5  # cells; check area around frontier
        
        # Sample some frontier cells and check their surroundings
        sample_size = min(10, len(frontier_cells))
        sampled_cells = list(frontier_cells)[:sample_size]
        
        for r, c in sampled_cells:
            for dr in range(-check_radius, check_radius + 1):
                for dc in range(-check_radius, check_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        total_checked += 1
                        cell_val = int(map_array[nr, nc])
                        if cell_val == -1:  # unknown
                            unknown_count += 1
        
        if total_checked == 0:
            return 0.0
        
        return unknown_count / total_checked
    
    def _check_area_stagnation(self):
        """
        Check if robot has been in the same area for too long.
        If so, clear nearby visited frontiers to force exploration elsewhere.
        """
        if self.robot_position is None:
            return
        
        now = time.time()
        robot_row, robot_col = self.robot_position
        
        # Check if robot has moved significantly from last area position
        if self.last_area_position is None:
            self.last_area_position = self.robot_position
            self.last_area_change_time = now
            return
        
        last_row, last_col = self.last_area_position
        distance = np.sqrt((robot_row - last_row)**2 + (robot_col - last_col)**2)
        
        if distance > self.area_stagnation_radius:
            # Robot has moved to a new area - reset timer
            self.last_area_position = self.robot_position
            self.last_area_change_time = now
        else:
            # Robot is still in same area - check if it's been too long
            elapsed = now - self.last_area_change_time
            if elapsed > self.area_stagnation_timeout:
                # Force exploration elsewhere by clearing nearby visited frontiers
                self.get_logger().warning(
                    f"Area stagnation detected! Robot in same area for {elapsed:.1f}s. "
                    f"Clearing nearby visited frontiers to force exploration elsewhere."
                )
                # Clear visited frontiers near current position
                to_remove = []
                for centroid in self.visited_frontiers:
                    dist = np.sqrt((centroid[0] - robot_row)**2 + (centroid[1] - robot_col)**2)
                    if dist < self.area_stagnation_radius * 2:  # Clear larger area
                        to_remove.append(centroid)
                for centroid in to_remove:
                    self.visited_frontiers.discard(centroid)
                if to_remove:
                    self.get_logger().debug(f"Cleared {len(to_remove)} nearby visited frontiers")
                # Reset stagnation timer
                self.last_area_change_time = now
    
    def get_recently_explored_penalty(self, centroid, map_array):
        """
        Calculate penalty for frontiers near recently explored areas.
        Returns a multiplier (0.0-1.0) where 1.0 = no penalty, lower = more penalty.
        """
        if not self.recently_explored_areas:
            return 1.0
        
        now = time.time()
        robot_row, robot_col = self.robot_position
        centroid_row, centroid_col = centroid
        
        # Clean up old explored areas
        self.recently_explored_areas = [
            (r, c, t) for r, c, t in self.recently_explored_areas
            if (now - t) < self.explored_area_retention_time
        ]
        
        if not self.recently_explored_areas:
            return 1.0
        
        # Check distance to recently explored areas
        min_distance = float('inf')
        for explored_row, explored_col, _ in self.recently_explored_areas:
            dist = np.sqrt(
                (centroid_row - explored_row)**2 + (centroid_col - explored_col)**2
            )
            min_distance = min(min_distance, dist)
        
        # Apply penalty if too close to recently explored area
        if min_distance < self.explored_area_penalty_radius:
            # Linear penalty: 0.0 at distance 0, 1.0 at penalty_radius
            penalty = min_distance / self.explored_area_penalty_radius
            # Make penalty more aggressive (quadratic)
            penalty = penalty ** 2
            return max(0.1, penalty)  # Minimum 10% of score
        
        return 1.0  # No penalty if far enough away
    
    def get_wall_proximity_ratio(self, frontier_cells, map_array):
        """
        Compute the fraction of frontier cells that are 'near' an occupied cell.
        Returns a ratio in [0,1] where 1 means all frontier cells are adjacent
        (within `wall_proximity_distance`) to an occupied cell.
        """
        if not frontier_cells:
            return 0.0

        rows, cols = map_array.shape
        near_count = 0
        pd = int(max(1, self.wall_proximity_distance))

        for r, c in frontier_cells:
            found = False
            for dr in range(-pd, pd + 1):
                for dc in range(-pd, pd + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if int(map_array[nr, nc]) >= 50:
                        found = True
                        break
                if found:
                    break
            if found:
                near_count += 1

        return near_count / max(1, len(frontier_cells))

    def get_idle_penalty(self):
        """
        Compute penalty for standing idle too long.
        Returns a multiplier (0.0-1.0) where 1.0 = no penalty, 0.0 = severe penalty.
        Encourages robot to explore instead of standing still.
        """
        now = time.time()
        idle_time = now - self.last_position_time
        
        # Calculate distance moved since last check (in map cells)
        robot_row, robot_col = self.robot_position
        distance = np.sqrt((robot_row - self.last_position[0])**2 + (robot_col - self.last_position[1])**2)
        
        # If moved significantly, reset idle timer
        if distance > self.idle_threshold_distance_cells:
            self.last_position = self.robot_position
            self.last_position_time = now
            return 1.0  # No penalty
        
        # Penalty increases linearly after idle_threshold_time
        if idle_time > self.idle_threshold_time:
            idle_penalty = max(0.1, 1.0 - (idle_time - self.idle_threshold_time) / 30.0)
            self.get_logger().warning(f"Idle penalty applied: {idle_penalty:.2f} (idle for {idle_time:.1f}s)")
            return idle_penalty
        
        return 1.0  # No penalty while idle time is acceptable

    def update_exploration_distance(self):
        """
        DISABLED: Breadcrumb tracking removed in favor of sensor fusion approach.
        The robot now relies on Cartographer's native loop closure detection
        combined with odometry/IMU fusion to maintain accuracy.
        """
        pass

    # --- Helper math utilities for transform math ---
    def _quat_to_mat(self, q):
        # q: geometry_msgs Quaternion-like tuple/list (x,y,z,w)
        x, y, z, w = q
        # normalize
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n == 0:
            return np.eye(3)
        x, y, z, w = x/n, y/n, z/n, w/n
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        m = np.array([
            [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
            [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
            [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)]
        ])
        return m

    def _mat_to_quat(self, m):
        # m: 3x3 rotation matrix -> returns (x,y,z,w)
        tr = m[0,0] + m[1,1] + m[2,2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (m[2,1] - m[1,2]) / S
            y = (m[0,2] - m[2,0]) / S
            z = (m[1,0] - m[0,1]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / S
            x = 0.25 * S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25 * S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25 * S
        return (x, y, z, w)

    def _send_map_to_odom_transform(self, trans_translation, trans_quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value if self.has_parameter('odom_frame') else 'odom'
        t.transform.translation.x = float(trans_translation[0])
        t.transform.translation.y = float(trans_translation[1])
        t.transform.translation.z = 0.0
        t.transform.rotation.x = float(trans_quat[0])
        t.transform.rotation.y = float(trans_quat[1])
        t.transform.rotation.z = float(trans_quat[2])
        t.transform.rotation.w = float(trans_quat[3])
        try:
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().debug(f"Failed to broadcast map->odom correction: {e}")

    def apply_odometry_correction(self):
        """
        Compute a temporary correction transform T_map_odom = T_map_base * inverse(T_odom_base)
        and publish it repeatedly during checkpoint validation. This effectively shifts
        the odom frame to align with the map at the robot pose, avoiding aggressive
        map merging.
        """
        map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        odom_frame = 'odom'
        try:
            now = rclpy.time.Time()
            t_map_base = self.tf_buffer.lookup_transform(map_frame, robot_frame, now)
            t_odom_base = self.tf_buffer.lookup_transform(odom_frame, robot_frame, now)
        except Exception as e:
            self.get_logger().debug(f"TF lookup for odom correction failed: {e}")
            return False

        # Extract translations and quaternions
        tm = t_map_base.transform.translation
        qm = t_map_base.transform.rotation
        to = t_odom_base.transform.translation
        qo = t_odom_base.transform.rotation

        tm_v = np.array([tm.x, tm.y, tm.z])
        to_v = np.array([to.x, to.y, to.z])
        qm_t = (qm.x, qm.y, qm.z, qm.w)
        qo_t = (qo.x, qo.y, qo.z, qo.w)

        Rm = self._quat_to_mat(qm_t)
        Ro = self._quat_to_mat(qo_t)

        # Compute R = Rm * Ro^{-1}
        R = Rm.dot(Ro.T)
        # Compute p = tm - R * to
        p = tm_v - R.dot(to_v)

        q_corr = self._mat_to_quat(R)

        # Publish the transform (map -> odom) so other nodes see corrected odom
        self._send_map_to_odom_transform(p[:2], q_corr)
        return True

    # --- SENSOR FUSION: Enhanced odometry and IMU integration ---
    
    def estimate_odometry_drift(self):
        """
        SENSOR FUSION: Estimate cumulative odometry drift based on velocity changes.
        Internally, tracks the difference between expected and observed movements.
        Returns the estimated drift (meters) as a confidence metric for other systems.
        """
        # In a real system, this would compare wheel encoder deltas vs actual displacement
        # For now, provide a simple estimate based on exploration time
        if self.last_odom_pose is None:
            self.last_odom_pose = self.robot_position
            return 0.0
        
        # Calculate recent movement
        row_delta = abs(self.robot_position[0] - self.last_odom_pose[0])
        col_delta = abs(self.robot_position[1] - self.last_odom_pose[1])
        recent_movement = np.sqrt(row_delta**2 + col_delta**2)
        
        # Update tracked position
        self.last_odom_pose = self.robot_position
        
        # Accumulated drift model: increases with distance traveled
        # This is a simplified model; real implementation would use encoder/visual odometry comparison
        drift_rate = 0.02  # 2% drift per meter (typical for wheel odometry)
        estimated_new_drift = self.odom_drift_estimate + (recent_movement * drift_rate)
        
        # Cap drift estimate to prevent unrealistic values
        self.odom_drift_estimate = min(estimated_new_drift, 5.0)  # Max 5 meters assumed drift
        
        return self.odom_drift_estimate

    def extract_lidar_features(self, map_array):
        """
        FEATURE EXTRACTION: Identify stable environmental features from LiDAR data.
        Returns a list of detected features with properties for loop closure matching.
        """
        # Simple feature detection: find edges (obstacles) in the map
        # In a production system, this would use corner detection, edge extraction, etc.
        features = []
        rows, cols = map_array.shape
        
        # Detect corners and edges by looking for high gradient (obstacle boundaries)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                cell_val = int(map_array[r, c])
                
                # Look for occupied cells next to free cells (boundaries)
                if cell_val >= 50:  # occupied
                    # Check neighbors
                    has_free_neighbor = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                neighbor_val = int(map_array[nr, nc])
                                if neighbor_val == 0:  # free cell
                                    has_free_neighbor = True
                                    break
                        if has_free_neighbor:
                            break
                    
                    if has_free_neighbor:
                        # This is an edge feature
                        features.append({
                            'row': r,
                            'col': c,
                            'type': 'edge',
                            'confidence': 0.8
                        })
        
        return features[:50]  # Limit to 50 most relevant features

    def match_features_for_loop_closure(self, current_features, stored_features):
        """
        LOOP CLOSURE DETECTION: Match current LiDAR features against stored observations.
        Returns a confidence score (0.0-1.0) indicating loop closure likelihood.
        """
        if not current_features or not stored_features:
            return 0.0
        
        # Simple matching: count nearby features (spatial proximity)
        matches = 0
        threshold_distance = 5  # cells; features within this distance count as match
        
        for curr_feat in current_features:
            for stored_feat in stored_features:
                dist = np.sqrt(
                    (curr_feat['row'] - stored_feat['row'])**2 +
                    (curr_feat['col'] - stored_feat['col'])**2
                )
                if dist < threshold_distance:
                    matches += 1
                    break
        
        # Confidence = (matched features / total features)
        confidence = min(1.0, matches / max(1, len(current_features)))
        return confidence

    def compute_fused_odometry_weight(self):
        """
        SENSOR FUSION: Dynamically adjust odometry weight based on drift estimation and loop closure.
        Returns adjusted weight for odometry fusion (0.0-1.0).
        """
        base_weight = 0.7
        
        # Reduce odometry weight as drift accumulates
        drift_penalty = self.odom_drift_estimate / 10.0  # Normalize drift (0-10m -> 0.0-1.0)
        adjusted_weight = max(0.3, base_weight - drift_penalty)
        
        # If loop closure was recently detected, reduce odom weight temporarily
        # to favor SLAM map-frame readings
        if self.loop_closure_detected:
            now = time.time()
            if (now - self.last_loop_closure_time) < 5.0:  # Within 5 seconds of loop closure
                adjusted_weight *= 0.5
                self.loop_closure_detected = False  # Clear flag after use
        
        self.odom_weight = adjusted_weight
        self.imu_weight = 1.0 - adjusted_weight  # Complementary weighting
        
        return self.odom_weight

    def _check_if_stuck(self):
        """
        Check if robot has a goal but isn't moving toward it.
        If stuck for too long, clear the goal to force a new one.
        """
        if self.current_goal is None or self.current_goal == "drift_correction":
            return
        
        if self.robot_position is None or self.last_movement_position is None:
            return
        
        now = time.time()
        # Only check movement periodically to avoid excessive computation
        if now - self.last_movement_check_time < self.movement_check_interval:
            return
        
        self.last_movement_check_time = now
        
        # Calculate distance moved since last check
        robot_row, robot_col = self.robot_position
        last_row, last_col = self.last_movement_position
        
        # Convert to meters (assuming 0.05m resolution)
        distance_moved = np.sqrt(
            ((robot_row - last_row) * 0.05)**2 + 
            ((robot_col - last_col) * 0.05)**2
        )
        
        # Check how long we've been stuck
        time_since_goal = now - self.last_goal_sent_time
        
        if distance_moved < self.stuck_movement_threshold:
            # Robot hasn't moved much
            if time_since_goal > self.stuck_timeout:
                # Been stuck for too long - clear goal and force new one
                self.get_logger().warning(
                    f"Robot stuck! Moved only {distance_moved:.2f}m in {time_since_goal:.1f}s. "
                    f"Clearing goal to force new exploration."
                )
                # Mark current goal as failed before clearing
                if isinstance(self.current_goal, tuple):
                    self.failed_frontiers[self.current_goal] = time.time()
                self.current_goal = None
        else:
            # Robot is moving - update last position
            self.last_movement_position = self.robot_position

    def log_sensor_fusion_state(self):
        """
        LOGGING: Report current sensor fusion state for debugging and analysis.
        Helps understand how much the system trusts odometry vs other sensors.
        """
        self.get_logger().debug(
            f"Sensor Fusion State: "
            f"odom_weight={self.odom_weight:.2f}, imu_weight={self.imu_weight:.2f}, "
            f"drift_estimate={self.odom_drift_estimate:.2f}m, "
            f"position={self.robot_position}"
        )
        """
        DISABLED: Breadcrumb-based return logic has been disabled.
        Cartographer now handles loop closure detection natively.
        The explorer node focuses on continuous frontier exploration with
        sensor fusion (odometry + IMU) to improve accuracy without forced returns.
        """
        return False, None, None

    def handle_checkpoint_arrival(self):
        """
        DISABLED: Checkpoint validation removed in favor of continuous sensor fusion.
        """
        return True

    def check_pause_timeout(self):
        """
        NEW: Detect if we've been paused waiting for Cartographer optimizations.
        Returns True if we should force movement despite low SLAM confidence.
        This is a safety valve to prevent endless waiting for perfection.
        """
        now = time.time()
        pause_duration = now - self.last_goal_sent_time
        
        # If no goal sent or we moved significantly, reset timer
        if self.current_goal is None:
            self.last_goal_sent_time = now
            self.consecutive_paused_iterations = 0
            return False
        
        # Increment counter if paused for multiple iterations
        if pause_duration > 0.8:  # After 0.8 seconds, consider it a pause (reduced for faster detection)
            self.consecutive_paused_iterations += 1
        else:
            self.consecutive_paused_iterations = 0
        
        # Force new frontier search after max_goal_pause_time seconds of pause
        if pause_duration > self.max_goal_pause_time:
            self.get_logger().warning(
                f"Pause timeout exceeded ({pause_duration:.1f}s): forcing new frontier "
                f"despite SLAM optimization. (iterations: {self.consecutive_paused_iterations})"
            )
            self.current_goal = None  # Clear current goal to trigger new frontier search
            self.last_goal_sent_time = now
            self.consecutive_paused_iterations = 0
            return True
        
        return False

    def choose_frontier(self, clusters, map_array):
        """
        Choose a frontier cluster to explore. Preference is given to larger
        clusters (more unknown area). Filters out unreachable frontiers and
        penalizes wall-hugging frontiers. Tiny clusters are deprioritized.
        Also penalizes all frontiers if robot has been idle too long.
        """
        robot_row, robot_col = self.robot_position

        now = time.time()

        best_score = -float('inf')
        chosen = None
        
        # Get idle penalty - applies to ALL frontiers if robot standing still
        idle_penalty = self.get_idle_penalty()

        # compute scores for clusters
        for cl in clusters:
            centroid = cl['centroid']
            size = cl['size']
            cells = cl['cells']

            # skip if we've visited this centroid already
            if centroid in self.visited_frontiers:
                continue

            # skip recently failed
            last_failed = self.failed_frontiers.get(centroid)
            if last_failed is not None and (now - last_failed) < 30.0:
                # skip clusters failed within last 30s (reduced from 60s for faster retry)
                continue

            # Skip frontiers that appear unreachable (behind walls/obstacles)
            if not self.is_frontier_reachable(cells, map_array):
                self.get_logger().debug(f"Frontier {centroid} appears unreachable, skipping")
                continue

            # distance (cell units)
            dist = np.sqrt((robot_row - centroid[0])**2 + (robot_col - centroid[1])**2)

            # Score: prefer larger clusters and closer ones.
            score = size / (1.0 + dist)

            # CRITICAL: Bonus for frontiers with high unknown space density
            # This incentivizes exploring areas that will expand the map more
            unknown_density = self.get_unknown_space_density(cells, map_array)
            unknown_bonus = 1.0 + (unknown_density * 2.0)  # Up to 3x bonus for high unknown density
            score *= unknown_bonus

            # CRITICAL: Penalty for frontiers near recently explored areas
            # This prevents the robot from staying in one place and revisiting same regions
            explored_penalty = self.get_recently_explored_penalty(centroid, map_array)
            score *= explored_penalty

            # small clusters get a penalty so isolated single-cell frontiers
            # are less likely to be chosen
            if size < self.min_frontier_size:
                score *= 0.3

            # Apply safety penalty (avoid dangerously close to obstacles)
            safety_mult = self.get_cluster_safety_score(cells, map_array)
            score *= safety_mult

            # Apply wall-proximity bonus: prefer frontiers that are near walls/outlines
            # This encourages the robot to trace boundaries before venturing into
            # large open areas where LiDAR provides poor constraints.
            proximity = self.get_wall_proximity_ratio(cells, map_array)
            wall_bonus = 1.0 + (self.wall_preference * proximity)
            score *= wall_bonus

            # CRITICAL: In low-feature areas, strongly prefer frontiers with more features
            # This helps avoid areas where localization might fail (like near rounded objects)
            if self.low_feature_area_detected:
                # Calculate feature density around this frontier
                feature_count = 0
                check_radius = 3
                for r, c in cells[:10]:  # Sample first 10 cells for speed
                    for dr in range(-check_radius, check_radius + 1):
                        for dc in range(-check_radius, check_radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < map_array.shape[0] and 0 <= nc < map_array.shape[1]:
                                if int(map_array[nr, nc]) >= 50:
                                    feature_count += 1
                # Boost score for frontiers with more features nearby
                feature_bonus = 1.0 + (feature_count / 100.0)  # Small bonus per feature
                score *= feature_bonus
            
            # CRITICAL: After drift recovery, strongly prefer frontiers far from recovery location
            # This encourages exploration of new areas rather than getting stuck refining
            if self.in_post_recovery_mode and self.post_recovery_position is not None:
                robot_row, robot_col = self.robot_position
                recovery_row, recovery_col = self.post_recovery_position
                distance_from_recovery = np.sqrt(
                    (centroid[0] - recovery_row)**2 + (centroid[1] - recovery_col)**2
                )
                # Strongly prefer frontiers that are far from where recovery happened
                # This forces exploration of new areas instead of staying stuck
                distance_bonus = 1.0 + (distance_from_recovery / 50.0)  # Bonus increases with distance
                score *= distance_bonus
                self.get_logger().warning(
                    f"Post-recovery: Frontier {centroid} distance from recovery: {distance_from_recovery:.1f} cells, "
                    f"bonus: {distance_bonus:.2f}"
                )

            # Apply idle penalty - if standing still, strongly encourage ANY frontier
            score *= idle_penalty
            
            # Sensor-fusion mode: no breadcrumb loop-closure bonus here.
            # Odometry/IMU and Cartographer will handle loop closure natively.

            if score > best_score:
                best_score = score
                chosen = cl

        if chosen:
            centroid = chosen['centroid']
            self.visited_frontiers.add(centroid)
            
            # Track this area as recently explored to avoid revisiting
            if self.robot_position is not None:
                robot_row, robot_col = self.robot_position
                self.recently_explored_areas.append((robot_row, robot_col, time.time()))
                # Limit list size to prevent memory growth
                if len(self.recently_explored_areas) > 100:
                    self.recently_explored_areas.pop(0)
            
            safety = self.get_cluster_safety_score(chosen['cells'], map_array)
            unknown_density = self.get_unknown_space_density(chosen['cells'], map_array)
            explored_penalty = self.get_recently_explored_penalty(centroid, map_array)
            self.get_logger().debug(
                f"Chosen frontier centroid: {centroid} size={chosen['size']} "
                f"safety_score={safety:.2f} idle_penalty={idle_penalty:.2f} "
                f"unknown_density={unknown_density:.3f} explored_penalty={explored_penalty:.2f} "
                f"robot_position={self.robot_position}"
            )
            return chosen
        else:
            self.get_logger().warning("No valid frontier cluster found")
            return None

    def explore(self):
        if self.map_data is None:
            self.get_logger().debug("No map data available")
            return

        # Monitor map stability to detect sudden corrections/jumps
        self.monitor_map_stability()
        
        # CRITICAL: If drift detected, actively correct it by navigating to known location
        if self.drift_recovery_mode:
            # Check if we've been correcting for too long
            if self.drift_correction_active:
                elapsed = time.time() - self.drift_correction_start_time
                if elapsed > self.drift_correction_timeout:
                    self.get_logger().warning(
                        f"Drift correction timeout ({elapsed:.1f}s). "
                        f"Abandoning correction and resuming exploration."
                    )
                    self.drift_correction_active = False
                    self.drift_recovery_mode = False
                    self.drift_correction_goal = None
                    # Clear current goal
                    if self.current_goal is not None:
                        self.current_goal = None
                    return
            
            # If we have a correction goal, navigate to it
            if self.drift_correction_goal is not None:
                goal_x, goal_y = self.drift_correction_goal
                # Check if we're already navigating to this goal
                if self.current_goal is None:
                    self.get_logger().info(
                        f"Drift correction: Navigating to goal ({goal_x:.2f}, {goal_y:.2f})"
                    )
                    self.navigate_to(goal_x, goal_y)
                    # Mark this as current goal (use a special marker)
                    self.current_goal = "drift_correction"
                return
            else:
                # Still calculating goal - add timeout to prevent infinite waiting
                elapsed = time.time() - self.drift_correction_start_time
                if elapsed > 3.0:  # If calculating goal for more than 3 seconds, give up
                    self.get_logger().warning(
                        f"Drift correction goal calculation timeout ({elapsed:.1f}s). "
                        f"Abandoning drift correction and resuming exploration."
                    )
                    self.drift_correction_active = False
                    self.drift_recovery_mode = False
                    self.drift_correction_goal = None
                    if self.current_goal is not None:
                        self.current_goal = None
                else:
                    self.get_logger().debug("Calculating drift correction goal...")
                return
        
        # Post-recovery check: if robot is stuck doing small movements after recovery, force exploration
        if self.in_post_recovery_mode:
            now = time.time()
            elapsed = now - self.drift_recovery_time
            
            if elapsed > self.post_recovery_timeout:
                # Check if robot has moved significantly since recovery
                if self.post_recovery_position is not None and self.robot_position is not None:
                    distance = np.sqrt(
                        (self.robot_position[0] - self.post_recovery_position[0])**2 +
                        (self.robot_position[1] - self.post_recovery_position[1])**2
                    )
                    
                    if distance < self.post_recovery_movement_threshold:
                        # Robot is stuck refining - force it to explore elsewhere
                        self.get_logger().warning(
                            f"Post-recovery timeout: Robot stuck refining localization "
                            f"(moved only {distance:.2f}m in {elapsed:.1f}s). "
                            f"Forcing exploration of new areas."
                        )
                        # Clear current goal and visited frontiers near current position
                        if self.current_goal is not None:
                            self.current_goal = None
                        # Mark nearby frontiers as visited to force exploration elsewhere
                        if self.robot_position is not None:
                            robot_row, robot_col = self.robot_position
                            nearby_threshold = 20  # cells (~1m at 0.05m resolution)
                            to_remove = []
                            for centroid in self.visited_frontiers:
                                dist = np.sqrt((centroid[0] - robot_row)**2 + (centroid[1] - robot_col)**2)
                                if dist < nearby_threshold:
                                    to_remove.append(centroid)
                            for centroid in to_remove:
                                self.visited_frontiers.discard(centroid)
                            if to_remove:
                                self.get_logger().debug(f"Cleared {len(to_remove)} nearby frontiers to force exploration elsewhere")
                        self.in_post_recovery_mode = False
                    else:
                        # Robot has moved enough, exit post-recovery mode
                        self.get_logger().debug(
                            f"Post-recovery: Robot moved {distance:.2f}m. Resuming normal exploration."
                        )
                        self.in_post_recovery_mode = False
                else:
                    # Can't check movement, just exit post-recovery mode after timeout
                    self.in_post_recovery_mode = False
        
        # If map just jumped (loop closure correction), pause exploration briefly
        if not self.is_map_stable():
            self.get_logger().debug("Waiting for map to stabilize after loop closure correction...")
            return

        # Update robot position from TF
        self.update_robot_position_from_tf()
        
        # Check if robot is stuck (not moving toward goal)
        # This is separate from pause timeout - it detects actual lack of movement
        self._check_if_stuck()
        
        # DISABLED: Breadcrumb distance tracking and checkpoint handling
        # Now relying on Cartographer's native loop closure and sensor fusion
        # self.update_exploration_distance()
        # if self.returning_to_checkpoint:
        #     ready_to_explore = self.handle_checkpoint_arrival()
        #     if not ready_to_explore:
        #         return
        # must_return, return_x, return_y = self.should_return_for_loop_closure()
        # if must_return:
        #     self.navigate_to(return_x, return_y)
        #     return

        # Convert map to numpy array (ensure dtype keeps -1 for unknown)
        try:
            raw = np.array(self.map_data.data, dtype=np.int8)
            expected = int(self.map_data.info.height) * int(self.map_data.info.width)
            if raw.size != expected:
                self.get_logger().error(f"Map data length mismatch: {raw.size} != {expected}")
                return
            map_array = raw.reshape((self.map_data.info.height, self.map_data.info.width))
        except Exception as e:
            self.get_logger().error(f"Failed to construct map array: {e}")
            return

        # CRITICAL: Detect low-feature areas (like near rounded objects)
        # This helps identify when we should rely more on IMU/odometry
        is_low_feature = self.detect_low_feature_area(map_array)
        
        if is_low_feature:
            # In low-feature areas, be more conservative with navigation
            # Prefer frontiers that are closer and have more features nearby
            self.get_logger().debug("In low-feature area - using conservative navigation")

        # Detect frontier clusters
        clusters = self.find_frontiers(map_array)

        if not clusters:
            # self.get_logger().debug("No frontiers found. Exploration complete!")
            return

        # CRITICAL: Don't send a new goal if we already have an active goal
        # This prevents spamming Nav2 and allows current navigation to complete
        if self.current_goal is not None:
            # Check if we've been stuck/paused too long - if so, clear goal to force new one
            if not self.check_pause_timeout():
                # Goal is still active and not timed out, skip sending new goal
                return
        
        # Check if robot has been in same area too long - force exploration elsewhere
        self._check_area_stagnation()
        
        # Choose the best frontier cluster (now includes reachability check)
        chosen_cluster = self.choose_frontier(clusters, map_array)

        if not chosen_cluster:
            self.get_logger().debug("No frontiers to explore")
            return

        centroid = chosen_cluster['centroid']
        # store current goal so failures can be recorded
        self.current_goal = centroid
        self.last_goal_sent_time = time.time()  # NEW: Reset pause timer when sending new goal

        # Convert the chosen frontier centroid to world coordinates
        goal_x = centroid[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_y = centroid[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y

        # Navigate to the chosen frontier centroid
        self.navigate_to(goal_x, goal_y)


def main(args=None):
    rclpy.init(args=args)
    explorer_node = ExplorerNode()

    try:
        explorer_node.get_logger().debug("Starting exploration...")
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info("Exploration stopped by user")
    finally:
        explorer_node.destroy_node()
        rclpy.shutdown()
