-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- /* Author: Darby Lim */

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",  -- Use imu_link since IMU topic exists and provides better orientation
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  -- BALANCED: Use odometry for speed BUT fuse with IMU for accuracy
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.5,  -- Increased to handle TF timing issues
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 0.02,  -- Increased to reduce TF timing issues and improve stability
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- Improved 2D trajectory builder settings for straighter wall detection
TRAJECTORY_BUILDER_2D.min_range = 0.12
TRAJECTORY_BUILDER_2D.max_range = 3.5
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.
-- CRITICAL: Enable IMU data for sensor fusion to correct odometry drift
-- IMU provides direct orientation measurements which help maintain correct angle
-- especially in featureless areas where LiDAR can't see features
-- In low-feature areas (like near rounded objects), IMU becomes critical for maintaining orientation
TRAJECTORY_BUILDER_2D.use_imu_data = true  -- Enabled - IMU topic exists and will help with orientation
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true

-- TIGHT: Motion filter to detect orientation changes quickly
-- Critical for curved paths - need to update frequently to track orientation changes
-- Tighter thresholds help maintain orientation accuracy during turns
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.3)  -- Tighter to catch orientation changes faster
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.2  -- Tighter to maintain accuracy during curves

-- CRITICAL: Better handling of featureless areas (open spaces)
-- When LiDAR can't see features, rely more on odometry
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90  -- More scans per submap for better matching
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

-- CRITICAL: Add constraints to prevent orientation drift in featureless areas
-- When max_range is exceeded, maintain orientation from odometry
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.5  -- Match max_range to prevent extrapolation
TRAJECTORY_BUILDER_2D.max_range = 3.5  -- Ensure consistency

-- CRITICAL: Increase submap resolution and constraints for better orientation stability
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.miss_probability = 0.49
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.insert_free_space = true

-- CRITICAL: Very high rotation weight to maintain correct orientation
-- Essential for tracking around curved objects where orientation can drift
-- Higher weight prevents orientation confusion when following circular paths
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 3.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 50.0  -- Increased to 50.0 for curved object tracking
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.use_nonmonotonic_steps = true
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 50  -- Increased for better curved feature matching
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 1

-- CRITICAL: Improve scan matching quality for curved features
-- Use higher resolution matching to better detect circular objects
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.0
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- CONSERVATIVE: Higher thresholds to prevent false loop closures from similar walls
-- Higher scores = stronger matches required = fewer false positives
POSE_GRAPH.constraint_builder.min_score = 0.80
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.85

-- Optimize less frequently to allow more data before corrections
-- Prevents premature corrections that can break the map
POSE_GRAPH.optimize_every_n_nodes = 50

-- Fine-tune loop closure detection for better global consistency
POSE_GRAPH.constraint_builder.log_matches = true  -- Enable to debug false matches

-- CRITICAL: Limit maximum correction distance to prevent large jumps
-- Prevents map from "breaking" when false loop closures occur
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0
POSE_GRAPH.constraint_builder.max_constraint_angle = math.rad(30.0)

-- CONSERVATIVE: Smaller search windows to prevent matching distant similar walls
-- Reduces chance of false matches from repetitive environments
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 7.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(15.0)

-- CRITICAL: Very high rotation weight for loop closure to prevent orientation drift
-- Essential when matching curved features - prevents false orientation matches
POSE_GRAPH.constraint_builder.ceres_scan_matcher.translation_weight = 5.0
POSE_GRAPH.constraint_builder.ceres_scan_matcher.rotation_weight = 80.0  -- Increased to 80.0 for curved object tracking

-- Limit iterations to prevent over-optimization that can cause map distortion
POSE_GRAPH.constraint_builder.max_constraint_builder_iterations = 200

-- CRITICAL: Add max allowed correction to prevent sudden map jumps
-- This prevents the map from "breaking" when loop closures are applied
POSE_GRAPH.max_num_final_iterations = 50
POSE_GRAPH.global_sampling_ratio = 0.03

return options
