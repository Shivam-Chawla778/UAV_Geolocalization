import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
import numpy as np
import math
import cv2
from dronekit import connect
import time
from geopy.distance import distance as geo_distance
from geopy import Point
import logging
from collections import deque
import threading
import pyrealsense2 as rs
from ultralytics import YOLO
import csv  # <-- ADDED: Import the csv module

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class D455YoloHumanPositioning:
    def __init__(self, connection_string='/dev/ttyUSB0', baud=57600):
        # Camera parameters for Intel RealSense D455
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        
        # Intrinsics
        self.K = np.array([[448.44050858, 0, 302.36894562],
                          [0, 450.05835973, 244.72255502],
                          [0, 0, 1]], dtype=np.float64)
        # Distortion coefficients
        self.dist_coeffs = np.array([0.26974184, -1.56360967, -0.00950144,
                                    -0.00800682, 3.5658071], dtype=np.float64)
        
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        logger.info("Using provided camera calibration parameters")
        
        # YOLO Setup 
        logger.info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt') 
        self.PERSON_CLASS_ID = 0
        logger.info("YOLOv8 model loaded successfully.")

        self.vehicle = None
        self.pipeline = None
        self.config = None
        self.connection_string = connection_string
        self.baud = baud
        
        self.position_history = {}
        self.max_history = 10
        self.position_threshold = 50.0
        
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        
        # CSV Logging Setup 
        self.csv_file = None
        self.csv_writer = None
        self.log_file_name = f"drone_human_log_{int(time.time())}.csv"

        # Drone coordinate system (FRD - Forward, Right, Down)
        #   X_d -> Forward
        #   Y_d -> Right
        #   Z_d -> Down
        
        # Camera coordinate system 
        #   X_c -> To the right in the image
        #   Y_c -> Down in the image
        #   Z_c -> Looking out from the lens
        
        # cam_yaw=-drone_roll, cam_roll=-drone_yaw, cam_pitch=-90deg
    
        # Camera's X (image right) points along the Drone's -Y (left)
        # Camera's Y (image down) points along the Drone's +X (forward)
        # Camera's Z (out of lens) points along the Drone's +Z (down)
        # The columns of the R_cam_to_drone matrix are the camera's axes
        # expressed in the drone's coordinate system.
        cam_x_in_drone_coords = [0,1,0]   
        cam_y_in_drone_coords = [-1,0,0]   
        cam_z_in_drone_coords = [0,0,1]  
        
        self.R_cam_to_drone = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ]) 
        logger.info(f"Using fixed camera-to-drone rotation matrix:\n{self.R_cam_to_drone}")
        


    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        "Converts Euler angles (in radians) to a 3x3 rotation matrix."
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        
        # Standard aerospace sequence: Yaw, Pitch, Roll (Z-Y'-X'')
        R = R_yaw @ R_pitch @ R_roll
        return R
    

    def setup_realsense(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            logger.info("RealSense D455 initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RealSense: {e}")
            return False

    def get_color_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
            return None
        except Exception as e:
            logger.warning(f"Failed to get color frame: {e}")
            return None
    
    def pixel_to_camera_ray(self, u, v):
        try:
            point = np.array([[[u, v]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(point, self.K, self.dist_coeffs, P=self.K)
            u_corrected, v_corrected = undistorted[0, 0, 0], undistorted[0, 0, 1]
            x_norm = (u_corrected - self.cx) / self.fx
            y_norm = (v_corrected - self.cy) / self.fy
            ray_camera = np.array([x_norm, y_norm, 1.0])
            return ray_camera / np.linalg.norm(ray_camera), (u_corrected, v_corrected)
        except Exception as e:
            logger.error(f"Error in pixel to camera ray conversion: {e}")
            return None, None
    

    def calculate_ground_intersection_d455(self, u, v, R_cam_to_ned, altitude):
        
        try:
            ray_camera, corrected_coords = self.pixel_to_camera_ray(u, v)
            if ray_camera is None: return None, None, None
            # Rotate the ray from the camera's frame to the NED frame
            ray_ned = R_cam_to_ned @ ray_camera
            
            if abs(ray_ned[2]) < 1e-6: return None, None, None
            t = altitude / ray_ned[2]
            if t <= 0: return None, None, None
            intersection_ned = t * ray_ned
            return intersection_ned[0], intersection_ned[1], corrected_coords
        except Exception as e:
            logger.error(f"Error in D455 ground intersection calculation: {e}")
            return None, None, None

    def ned_to_latlon(self, lat_origin, lon_origin, north_offset, east_offset):
        R = 6378137.0
        lat_origin_rad = math.radians(lat_origin)
        lon_origin_rad = math.radians(lon_origin)
        lat_offset_rad = north_offset / R
        lon_offset_rad = east_offset / (R * math.cos(lat_origin_rad))
        target_lat = math.degrees(lat_origin_rad + lat_offset_rad)
        target_lon = math.degrees(lon_origin_rad + lon_offset_rad)
        return target_lat, target_lon
    
    def enhanced_position_filter(self, person_id, lat, lon):
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.max_history)
        if len(self.position_history[person_id]) > 2:
            recent_positions = list(self.position_history[person_id])[-3:]
            avg_lat = np.mean([pos[0] for pos in recent_positions])
            avg_lon = np.mean([pos[1] for pos in recent_positions])
            if geo_distance((avg_lat, avg_lon), (lat, lon)).meters > self.position_threshold:
                logger.warning(f"Rejecting outlier for person {person_id}")
                if self.position_history[person_id]:
                    return self.position_history[person_id][-1]
        self.position_history[person_id].append((lat, lon))
        positions = list(self.position_history[person_id])
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights /= weights.sum()
        avg_lat = sum(pos[0] * w for pos, w in zip(positions, weights))
        avg_lon = sum(pos[1] * w for pos, w in zip(positions, weights))
        return avg_lat, avg_lon
    
    # CSV logging
    def setup_csv_logging(self):
        
        try:
            
            self.csv_file = open(self.log_file_name, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # The header row
            header = [
                'timestamp', 'drone_lat', 'drone_lon', 'drone_alt', 'drone_heading',
                'human_id', 'human_lat_predicted', 'human_lon_predicted', 'distance_to_human_m'
            ]
            self.csv_writer.writerow(header)
            logger.info(f"CSV logging started. Data will be saved to {self.log_file_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up CSV logging: {e}")
            return False

    def log_to_csv(self, drone_lat, drone_lon, drone_alt, drone_heading, human_id, human_lat, human_lon, distance):
        """Writes a single row of data to the CSV file."""
        if self.csv_writer:
            try:
                timestamp = time.time()
                row = [
                    timestamp, drone_lat, drone_lon, drone_alt, drone_heading,
                    human_id, human_lat, human_lon, distance
                ]
                self.csv_writer.writerow(row)
            except Exception as e:
                logger.warning(f"Could not write to CSV file: {e}")

    def process_frame(self, frame):
        process_start = time.time()
        
        if not self.validate_telemetry():
            return frame
        
        # Drone's attitude from Pixhawk (Flight controller) 
        drone_roll = self.vehicle.attitude.roll
        drone_pitch = self.vehicle.attitude.pitch
        drone_yaw = self.vehicle.attitude.yaw # This is the heading
        
        # 1. Get the drone's attitude as a rotation matrix (from NED to Drone frame)
        R_ned_to_drone = self.euler_to_rotation_matrix(drone_roll, drone_pitch, drone_yaw)
        
        # We need the inverse: a matrix from Drone frame to NED frame
        R_drone_to_ned = R_ned_to_drone.T
        
        # 2. Combined with the fixed camera-to-drone rotation to get the final camera attitude
        # R_cam_to_ned = R_drone_to_ned @ R_cam_to_drone
        # It translates the ray vector from the camera's frame, to the drone's frame, to the world (NED) frame.
        R_cam_to_ned = R_drone_to_ned @ self.R_cam_to_drone
        

        results = self.yolo_model(frame, verbose=False)
        detected_humans = []
        altitude = self.vehicle.location.global_relative_frame.alt
        lat_origin = self.vehicle.location.global_frame.lat
        lon_origin = self.vehicle.location.global_frame.lon

        for r in results:
            for j, box in enumerate(r.boxes):
                if box.cls[0] == self.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    u_pixel, v_pixel = int((x1 + x2) / 2), int(y2)
                    
            
                    result = self.calculate_ground_intersection_d455(
                        u_pixel, v_pixel, R_cam_to_ned, altitude
                    )
            
                    
                    if result[0] is not None:
                        north_offset, east_offset, corrected_coords = result
                        target_lat, target_lon = self.ned_to_latlon(lat_origin, lon_origin, north_offset, east_offset)
                        filtered_lat, filtered_lon = self.enhanced_position_filter(j, target_lat, target_lon)
                        distance = math.sqrt(north_offset**2 + east_offset**2)
                        
                        detected_humans.append({
                            'id': j, 'distance': distance, 'lat': filtered_lat, 'lon': filtered_lon,
                            'north_offset': north_offset, 'east_offset': east_offset
                        })
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (u_pixel, v_pixel), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"P{j}: {distance:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # <-- ADDED: Call the logging function for each detected human -->
                        self.log_to_csv(
                            drone_lat=lat_origin,
                            drone_lon=lon_origin,
                            drone_alt=altitude,
                            drone_heading=math.degrees(drone_yaw), # Log heading in degrees
                            human_id=j,
                            human_lat=filtered_lat,
                            human_lon=filtered_lon,
                            distance=distance
                        )


        self.draw_enhanced_info(frame, detected_humans, drone_roll, drone_pitch, drone_yaw, altitude)
        
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        return frame
    
    def draw_enhanced_info(self, frame, detections, roll, pitch, yaw, altitude):
        y_offset = 30
        cv2.putText(frame, f"Alt: {altitude:.1f}m | IMU: Pixhawk", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"DRONE R:{math.degrees(roll):.1f} P:{math.degrees(pitch):.1f} Y:{math.degrees(yaw):.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        if self.processing_times:
            fps = 1.0 / np.mean(self.processing_times) if self.processing_times else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # rest of the info drawing 

    def validate_telemetry(self):
        if self.vehicle and self.vehicle.location.global_relative_frame.alt is not None and self.vehicle.location.global_frame.lat is not None:
            return True
        return False
    
    def connect_vehicle(self):
        try:
            self.vehicle = connect(self.connection_string, baud=self.baud, wait_ready=True)
            logger.info("Connected to vehicle successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vehicle: {e}")
            return False
    
    def run(self):
        # Setup CSV logging on run
        if not self.connect_vehicle() or not self.setup_realsense() or not self.setup_csv_logging():
            self.cleanup()
            return
            
        logger.info("Starting system. Press 'q' to quit.")
        try:
            while True:
                frame = self.get_color_frame()
                if frame is None: continue
                processed_frame = self.process_frame(frame)
                cv2.imshow("D455 YOLO Human Positioning System", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.csv_file:
            self.csv_file.close()
            logger.info("CSV log file closed.")
            
        if self.pipeline: self.pipeline.stop()
        cv2.destroyAllWindows()
        if self.vehicle: self.vehicle.close()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    positioning_system = D455YoloHumanPositioning(connection_string='tcp:127.0.0.1:5760')
    positioning_system.run()
