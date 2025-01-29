from VexAI import *

text = 'This is a demo text string. It is used to demonstrate the VexAI class.'
vex_ai = VexAI(text)

print("Testing next_word:")
print(vex_ai.next_word("demo"))

print("\nTesting sentence generation (length 10):")
user_input = "the"
print(vex_ai.sentence(user_input, 10))

print("\nTesting softmax function:")
x = [1.0, 2.0, 3.0]
print(vex_ai.softmax(x))

print("\nTesting matrix multiplication:")
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(vex_ai.matrix_multiplication(A, B))

print("\nTesting distance functions:")
print(vex_ai.distance(0, 0, 3, 4)) 
print(vex_ai.distance_3d(0, 0, 0, 1, 1, 1))
print(vex_ai.distance_1d(5, 10))

print("\nTesting midpoint functions:")
print(vex_ai.midpoint(0, 0, 3, 4))
print(vex_ai.midpoint_3d(0, 0, 0, 1, 1, 1))
print(vex_ai.midpoint_1d(5, 10))


print("\nTesting slope functions:")
print(vex_ai.slope(0, 0, 2, 2))
print(vex_ai.slope_3d(0, 0, 0, 1, 1, 1))

print("\nTesting pathfinding:")
print(vex_ai.pathfind(0, 0, 3, 3))

print("\nTesting triangulation:")
print(vex_ai.triangulate_location(0, 0, 2, 2, 4, 0))
print(vex_ai.triangulate_location_3d(0, 0, 0, 1, 1, 1, 2, 2, 2))

print("\nTesting SLAM:")
print(vex_ai.SLAM(0, 0, 5, 3, 0, 0))
print(vex_ai.SLAM_3d(0, 0, 0, 5, 3, 0, 0))

print("\nTesting Vision:")
coordinates = [(1, 1), (2, 2), (3, 3)]
map_grid, obstacles = vex_ai.Vision(coordinates)
print("Map Grid:")
for row in map_grid:
    print(row)
print("Obstacles:", obstacles)
