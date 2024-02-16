import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

# Function to calculate the Hamming distance between two arrays
def hamming_distance(arr1, arr2):
    # Resize arrays to match shape
    max_shape = (max(arr1.shape[0], arr2.shape[0]), max(arr1.shape[1], arr2.shape[1]))
    resized_arr1 = np.zeros(max_shape)
    resized_arr2 = np.zeros(max_shape)
    resized_arr1[:arr1.shape[0], :arr1.shape[1]] = arr1
    resized_arr2[:arr2.shape[0], :arr2.shape[1]] = arr2
    # Calculate Hamming distance
    return np.count_nonzero(resized_arr1 != resized_arr2)

# Function to calculate the percentage difference between two arrays
def percentage_difference(arr1, arr2):
    total_elements = max(arr1.size, arr2.size)
    differing_elements = hamming_distance(arr1, arr2)
    return 1-(differing_elements / total_elements)

# Function to check if a point is inside the polygon
def point_inside_polygon(point, polygon_coords):
    polygon = Polygon(polygon_coords)
    point = Point(point)
    return polygon.contains(point)

# Function to rotate a point around another point
def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return [qx, qy]

def load_polygon_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extracting the first polygon's coordinates
    return data[0]["ehitis"]["ehitiseKujud"]["ruumikuju"][0]["geometry"]["coordinates"][0]

# Function to check if the unit square is completely filled by the polygon
def is_square_filled_by_polygon(polygon_coords, square):
    for vertex in square:
        if not point_inside_polygon(vertex, polygon_coords):
            return False
    return True

# Function to calculate the angle to rotate the polygon
def angle_to_rotate_polygon(coords):
    max_length = 0
    angle_of_longest_edge = 0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if length > max_length:
            max_length = length
            angle_of_longest_edge = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0]))
    return -angle_of_longest_edge

# Function to calculate the area of intersection between two polygons
def intersection_area(poly1, poly2):
    intersection = poly1.intersection(poly2)
    return intersection.area

# Function to calculate the area of a polygon
def polygon_area(polygon_coords):
    return Polygon(polygon_coords).area

# Function to calculate the ratio of intersection area to total filled area
def calculate_coverage_ratio(polygon_coords, square_vertices):
    polygon = Polygon(polygon_coords)
    square_polygon = Polygon(square_vertices)
    intersection_area_value = intersection_area(polygon, square_polygon)
    square_area = polygon_area(polygon)  # Area of the square
    return intersection_area_value / square_area  # Divide by square area instead

# Modify your existing code inside process_multiple_json_files function
def process_multiple_json_files(file_paths):
    first_polygon_array = None
    for file_path in file_paths:
        polygon_coords = load_polygon_from_json(file_path)
        angle = angle_to_rotate_polygon(polygon_coords)
        origin = polygon_coords[0]
        rotated_polygon_coords = [rotate_point(origin, point, angle) for point in polygon_coords]
        min_x_rotated = min(coord[0] for coord in rotated_polygon_coords)
        min_y_rotated = min(coord[1] for coord in rotated_polygon_coords)

        transformed_rotated_coords = [[x - min_x_rotated, y - min_y_rotated] for x, y in rotated_polygon_coords]

        # Find highest point on y-axis and longest point on x-axis
        max_y = max(coord[1] for coord in transformed_rotated_coords)
        max_x = max(coord[0] for coord in transformed_rotated_coords)

        # Plotting
        plt.figure()
        plt.fill(*zip(*transformed_rotated_coords), 'b', alpha=0.5)
        plt.plot(*zip(*transformed_rotated_coords), 'r-')

        total = 0
        boxes = []

        for j in range(0, accuracy*int(max_x)+accuracy):
            temp = []
            for i in range(0, accuracy*int(max_y)+accuracy):
              # Calculate coverage ratio
              square_vertices = [(j/accuracy, i/accuracy), (j/accuracy, (1+i)/accuracy), ((1+j)/accuracy, (1+i)/accuracy), ((1+j)/accuracy, i/accuracy), (j/accuracy, i/accuracy)]
              coverage_ratio = calculate_coverage_ratio(square_vertices, transformed_rotated_coords[:-1])
              if (coverage_ratio >= 0.8):
                temp.append(1)
                total += 1
              else:
                temp.append(0)

              # Plotting the square
              plt.plot(*zip(*square_vertices), 'g--')  # Plotting square in green dashed lines

            boxes.append(temp)

        # Convert the boxes list into a numpy array
        polygon_array = np.array(boxes)

        # Store the array from the first polygon for comparison
        if first_polygon_array is None:
            first_polygon_array = polygon_array

        diff_percentage = percentage_difference(first_polygon_array, polygon_array)

        # Extracting file name from the file path
        file_name = file_path.split('.')[0]
        plt.title(f"Top-down view of the building EHR:{file_name}")  # Including file name in title
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        print(diff_percentage)
        print(total)

accuracy = 1

# List of JSON file paths
json_files = ["101017167.ehr.json",
              "101017568.ehr.json",
              "101020574.ehr.json",
              "101018823.ehr.json",
              "101019690.ehr.json",
              "101020222.ehr.json",
              "101014498.ehr.json",
              "101014817.ehr.json",
              "101023186.ehr.json",
              "101021013.ehr.json",
              "101021043.ehr.json",
              "101025957.ehr.json"]  # Replace with your file names


# Process multiple JSON files
process_multiple_json_files(json_files)
