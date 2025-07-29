import numpy as np
import math
import cv2
# Load ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
from dronekit import connect
import time
import math

# Connection string - adjust as needed
vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True)
print("Connected to vehicle")


def get_heading_deg(yaw_rad):
    # Convert radians to degrees and normalize
    heading = math.degrees(yaw_rad)
    heading = (heading + 360) % 360
    return heading


def calculate_ground_scale(camera_height, image_width, image_height, hfov_degrees):
    """
    Calculate the real-world scale (meters per pixel) for ground-facing camera
    
    Args:
        camera_height: Height above ground in meters
        image_width, image_height: Image dimensions in pixels  
        hfov_degrees: Horizontal field of view in degrees
    """

    # Calculate ground coverage
    hfov_rad = math.radians(hfov_degrees)
    vfov_rad = hfov_rad * (image_height / image_width)  # Assuming square pixels
    
    # Ground area covered by camera
    ground_width = 2 * camera_height * math.tan(hfov_rad / 2)
    ground_height = 2 * camera_height * math.tan(vfov_rad / 2)
    
    # Meters per pixel
    meters_per_pixel_x = ground_width / image_width
    meters_per_pixel_y = ground_height / image_height
    
    return meters_per_pixel_x, meters_per_pixel_y

def pixels_to_ground_distance_simple(pixel1, pixel2, meters_per_pixel_x, meters_per_pixel_y):
    """
    Calculate ground distance between two pixels
    """
    dx_pixels = pixel2[0] - pixel1[0]
    dy_pixels = pixel2[1] - pixel1[1]
    
    # Convert to real-world distance
    dx_meters = dx_pixels * meters_per_pixel_x
    dy_meters = dy_pixels * meters_per_pixel_y
    
    #Distance in meters
    distance = math.sqrt(dx_meters**2 + dy_meters**2)
    
    return distance

def calculate_marker_gps(pixel1, pixel2, meters_per_pixel_x, meters_per_pixel_y, vehicle_lat, vehicle_lon, drone_heading):
    """
    Calculate GPS coordinates of ArUco marker based on pixel displacement
    """
    dx_pixels = pixel2[0] - pixel1[0]
    dy_pixels = pixel2[1] - pixel1[1]
    
    # Convert to real-world distance
    dx_meters = dx_pixels * meters_per_pixel_x
    dy_meters = dy_pixels * meters_per_pixel_y
    
    # Calculate bearing from drone to marker (in image coordinates)
    bearing_rad = math.atan2(dx_meters, -dy_meters)  # Note: -dy because image y increases downward
    bearing_deg = math.degrees(bearing_rad)
    
    # Convert to world bearing (add drone heading)
    world_bearing = (drone_heading + bearing_deg) % 360
    world_bearing_rad = math.radians(world_bearing)
    
    # Distance to marker
    distance = math.sqrt(dx_meters**2 + dy_meters**2)
    
    # Earth radius in meters
    R = 6378137.0
    
    # Calculate lat/lon offsets
    lat_offset = (distance * math.cos(world_bearing_rad)) / R
    lon_offset = (distance * math.sin(world_bearing_rad)) / (R * math.cos(math.radians(vehicle_lat)))
    
    # Convert to degrees and add to vehicle position
    marker_lat = vehicle_lat + math.degrees(lat_offset)
    marker_lon = vehicle_lon + math.degrees(lon_offset)
    
    return marker_lat, marker_lon

# D455 typical specs (verify with your unit)
D455_HFOV = 87  # degrees (horizontal field of view for RGB)
D455_VFOV = 58  # degrees (vertical field of view for RGB)
IMAGE_WIDTH = 640   #  resolution
IMAGE_HEIGHT = 480  # resolution

# Camera matrix and distortion coefficients
camera_matrix = np.array([[640, 0, 320],
                         [0, 640, 240],
                         [0, 0, 1]], dtype=np.float32)

# Distortion coefficients (k1, k2, p1, p2, k3)
dist_coeffs = np.array([0.26974184, -1.56360967, -0.00950144,
                        -0.00800682, 3.5658071])

# Your setup
altitude = vehicle.location.global_relative_frame.alt  # meters (your known height)
camera_height = altitude  

# Calculate scale
scale_x, scale_y = calculate_ground_scale(
    camera_height, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT, 
    D455_HFOV
)

print(f"Scale: {scale_x:.4f} m/pixel (X), {scale_y:.4f} m/pixel (Y)")

# Start video capture
cap = cv2.VideoCapture(7)  # Video capture device index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
            # marker_corners is a 4x1x2 array -> reshape to 4x2
            corners_2d = marker_corners.reshape((4, 2))
            
            # Apply distortion correction
            corners_undistorted = cv2.undistortPoints(corners_2d.reshape(-1,1,2), camera_matrix, dist_coeffs, P=camera_matrix)
            corners_2d = corners_undistorted.reshape(-1,2)

            # Calculate center point (mean of the 4 corners)
            aruco_center_x = int(corners_2d[:, 0].mean())
            aruco_center_y = int(corners_2d[:, 1].mean())

            # Draw marker and center point
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (aruco_center_x, aruco_center_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {ids[i][0]} ({aruco_center_x}, {aruco_center_y})", (aruco_center_x+10, aruco_center_y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Print to terminal
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Resolution: {width} x {height}")

            print(f"Marker ID {ids[i][0]} center: (x={aruco_center_x}, y={aruco_center_y})")
            pixel2=(aruco_center_x,aruco_center_y)  # Center of the detected aruco marker
    
            pixel1 = (319,239)  #  Center of the image pixel coordinates
            
            # Calculating ground distance
            
            distance = pixels_to_ground_distance_simple(pixel1, pixel2, scale_x, scale_y)
            
            print(f"Ground distance: {distance:.2f} meters")
            
           
            # GPS coordinates
            vehicle_lat = vehicle.location.global_frame.lat
            vehicle_lon = vehicle.location.global_frame.lon
           
            # Heading from vehicle.attitude.yaw (in radians)
            yaw = vehicle.attitude.yaw  # in radians
            drone_heading = get_heading_deg(yaw)

            print(f"Vehicle_Latitude: {vehicle_lat:.6f}, Vehicle_Longitude: {vehicle_lon:.6f}")
            print(f"Altitude: {altitude:.2f} m")
            print(f"Compass Heading: {drone_heading:.2f}°")
            
            # Calculate ArUco marker GPS coordinates
            marker_lat, marker_lon = calculate_marker_gps(pixel1, pixel2, scale_x, scale_y, vehicle_lat, vehicle_lon, drone_heading)
            print(f"Marker GPS: Latitude: {marker_lat:.6f}, Longitude: {marker_lon:.6f}")
            print("------")
            
    cv2.imshow("Aruco Marker Center Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()