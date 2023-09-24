# IMPORT FUNCTION FOR NUMERIC OPERATIONS
import numpy as np

# MINIMUM ACCURACY OF FOUND POINTS AND FUNCTION
epsilon = 10e-3

def F(x_array):
    """
    FUNCTION IN WHICH WE SHOULD FIND MINIMUM
    """
    return (x_array[0]**2 + x_array[1]**2 - 1)**2 - (x_array[0]**3-x_array[1])**2

def apply_F_to_Ndim_array(points, F):
    """
    MAKES AN ARRAY WITH VALUES OF FUNCTION IN GIVEN POINTS
    """
    return [F(i) for i in points]

def sort_indexes_to_squared_polinom(data):
    # WORK ONLY FOR 3 POINTS
    sort_by_indexes = np.argsort(data)
    sort_by_indexes = [sort_by_indexes[1], sort_by_indexes[0], sort_by_indexes[2]]
    return sort_by_indexes

def find_min_point_squared(points, function_data):
    """
    FIND MINIMUM IN SQUARED FUNCTION LIKE: Y=AX^2+BX+C
    ARGUMENT DATA SHOULD PASS LIKE:
        POINTS [X1, X2, X3], [Y1, Y2, Y3], [Z1, Z2, Z3], ... [N1, N2, N3]
        FUNCTION ARRAY [F1, F2, F3]
    """
    function_data = np.array(function_data)
    right_indexes = sort_indexes_to_squared_polinom(function_data)
    points = ((points)[[right_indexes]])
    function_data = ((function_data)[[right_indexes]]).tolist()[0]
    # PARSE FUNCTION DATA ARRAY INTO 3 FUNCTIONS IN 3 POINTS
    f1, f2, f3 = function_data
    # MAKE ARRAY WHICH WE SHOULD FILL
    answer = []
    # FIND MINIMA IN EVERY COORDINATE, USING FOR LOOP:
    for x1, x2, x3 in points.T:
        x1=x1[0]
        x2=x2[0]
        x3=x3[0]
        answer.append(0.5*((((x2**2)-(x3**2))*f1)+(((x3**2)-(x1**2))*f2)+(((x1**2)-(x2**2))*f3))/(((x2-x3)*f1)+((x3-x1)*f2)+((x1-x2)*f3)))
    return answer

def find_min_point(points, function):
    """
    FIND INDEX OF GIVEN POINTS IN WHICH FUNCTION REACHS IT'S MINIMA
    """
    return np.argmin([function(i) for i in points])

def find_norm(radius_vector):
    """
    FIND NORM (LENGTH) OF VECTOR
    """
    return np.linalg.norm(radius_vector)

def squared_interpolation_minimum(points, function, epsilon, delta, multiplier):
    """
    POINTS SHOULD PASS LIKE: [X1, Y1, ..., N1], [X2, Y2, ..., N2], [X3, Y3, ..., N3]
    FUNCTION SHOULD TAKE PARAMETERS LIKE [X, Y, ..., N]
    EPSILON IS MINIMUM ACCURACY THAT WE HAVE IN THIS FUNCTION TO GET ANSWER
    DELTA IS MAXIMUM REDUCTION BY ALL COORDINATES FROM START POINT
    MULTIPLIER IS VARIABLE THAT WE MULTIPLY BY DELTA TO MOVE END POINTS EVERY STEP:
     (X_RIGHT-DELTA*MULTIPLIER^I), (X_LEFT+DELTA*MULTIPLIER^I), WHERE 'I' IS STEP NUMBER
    """
    # CURRENT DELTA IS MAIN POINT SHIFT 
    current_delta = delta

    # GET FUNCTION VALUES FROM ARRAY OF GIVEN POINTS
    function_data = apply_F_to_Ndim_array(points, function)
    # TRY TO FIND MINIMUM FROM GIVEN POINTS
    minimum_point = find_min_point_squared(points, function_data)
    # UPDATE MINIMUM POINTS ARRAY WITH POINT WHICH WE FOUND
    points = np.concatenate((points, [minimum_point]), axis=0)
    # FIND POINT IN WHICH FUNCTION REACHES ITS MINIMA
    minimum_point = (points)[find_min_point(points, function)]
    # CHECK MAIN CONDITIONS TO STOP LOOP:
    #   1)IF FUNCTION DOESNT GROW WHEN WE TRY TO FIND LOWER POINT
    #   2)WHEN POINT DOESNT CHANGE
    while (np.abs(np.min(function_data) - function(minimum_point))>epsilon or 
            np.abs(find_norm(np.min(points, axis=0) - minimum_point))>epsilon):
        # FIND SHIFTED POINTS
        left_point = np.array([])
        right_point = np.array([])
        for i in minimum_point:
            # FIND DELTA FROM NUMBER OF THIS STEP (GEOMETRIC PROGRESSION)
            # DELTA = PREVIOUS_DELTA * SOME_MULTIPLIER THAT WE DEFINE IN FUNCTION ARGUMENTS
            current_delta *= multiplier
            # FIND 2 POINTS BY ADD OR TAKE OFF THIS DELTA
            left_point = np.append(left_point, i - current_delta)
            right_point = np.append(right_point, i + current_delta)
        # TRY TO FIND MINIMUM FROM THIS COMBINED POINTS AGAIN, BUT WITH SHIFTED POINTS
        points = np.array([left_point, minimum_point, right_point])
        # FIND FUNCTION VALUE IN THIS THREE POINTS
        function_data = apply_F_to_Ndim_array(points, function)
        # TRY TO FIND MINIMUM FROM THIS COMBINED POINTS AGAIN
        minimum_point = find_min_point_squared(points, function_data)
        # UPDATE MINIMUM POINTS ARRAT WITH POINTS WHICH WE FOUND
        points = np.concatenate((points, np.array([minimum_point])), axis=0)
        # FIND POINT IN WHICH FUNCTION REACHES ITS MINIMA
        minimum_point = (points)[find_min_point(points, function)]
    # RETURN GIVEN RESULT IF CONDITIONS STOP WORKING 
    return minimum_point

if __name__ == "__main__":
    # FIND MINIMAL POINT WITH START 3 POINTS: [0.5, 0], [0.12, 1.12], [-0.5, 2],
    # IN FUNCTION F WITH GIVEN MINIMUM ERROR OF EPSILON
    # WITH MAXIMUM DELTA FROM MIDDLE POINT = 1
    # AND BASE IN GEOMETRICAL PROGRESSION EQUALS 0.999
    point = squared_interpolation_minimum(np.array([[0.5, 0], [0.12, 1.12], [-0.5, 2]]), F, epsilon, 1, 0.99)
    # PRINT THIS INFORMATION IN FORMATTED-STRING, WITH ACCURACY 4 VALUES AFTER INTEGER VALUE
    print(f"Minimum point is: {[round(i, 4) for i in point]}; And function data in this point is: {F(point):.4f}")