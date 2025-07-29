import numpy as np
import math
import cv2
from dronekit import connect
import time
from geopy.distance import distance as geo_distance
from geopy import Point

# ----------------------- Connection to Pixhawk -----------------------
vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True)
print("Connected to vehicle")

# ----------------------- Camera Parameters -----------------------
D455_HFOV = 87  # degrees (horizontal field of view)
D455_VFOV = 58  # degrees (vertical field of view)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Compute focal lengths in pixels
fx = IMAGE_WIDTH / (2 * math.tan(math.radians(D455_HFOV) / 2))
fy = IMAGE_HEIGHT / (2 * math.tan(math.radians(D455_VFOV) / 2))
cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# ----------------------- ArUco Setup -----------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ----------------------- Helper Functions -----------------------

def get_heading_deg(yaw_rad):
    heading = math.degrees(yaw_rad)
    return (heading + 360) % 360

def get_ground_offset_from_pixel(u, v, pitch_rad, altitude, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_cam = np.array([x, y, 1.0])
    R_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad),  math.cos(pitch_rad)]
    ])
    ray_world = R_pitch @ ray_cam
    t = altitude / ray_world[2]
    ground_point = ray_world * t
    return ground_point[0], ground_point[1]

def offset_to_latlon(lat, lon, dx, dy):
    origin = Point(lat, lon)
    lat_offset = geo_distance(meters=dx).destination(origin, 0).latitude  # forward (N)
    final_point = geo_distance(meters=dy).destination((lat_offset, lon), 90)  # right (E)
    return final_point.latitude, final_point.longitude

# ----------------------- Video Capture -----------------------
cap = cv2.VideoCapture(7)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not cap.isOpened():
    print("Camera not found.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_corners in enumerate(corners):
            corners_2d = marker_corners.reshape((4, 2))
            center_x = int(corners_2d[:, 0].mean())
            center_y = int(corners_2d[:, 1].mean())
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

            altitude = vehicle.location.global_relative_frame.alt
            pitch_rad = vehicle.attitude.pitch
            yaw_rad = vehicle.attitude.yaw
            heading_deg = get_heading_deg(yaw_rad)
            lat, lon = vehicle.location.global_frame.lat, vehicle.location.global_frame.lon

            dx, dy = get_ground_offset_from_pixel(center_x, center_y, pitch_rad, altitude, K)
            corrected_distance = math.sqrt(dx**2 + dy**2)
            corrected_lat, corrected_lon = offset_to_latlon(lat, lon, dx, dy)

            print(f"\n--- Marker ID {ids[i][0]} ---")
            print(f"Center Pixel: ({center_x}, {center_y})")
            print(f"Pitch: {math.degrees(pitch_rad):.2f}°, Heading: {heading_deg:.2f}°")
            print(f"Offset: dx = {dx:.2f} m, dy = {dy:.2f} m")
            print(f"Corrected Ground Distance: {corrected_distance:.2f} m")
            print(f"Original GPS: ({lat:.6f}, {lon:.6f})")
            print(f"Estimated Target GPS: ({corrected_lat:.6f}, {corrected_lon:.6f})")
            print("-----------------------------")

    cv2.imshow("Aruco Marker Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
