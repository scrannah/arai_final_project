from controller import Robot
import numpy as np
import cv2
import heapq

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# CONSTANTS / FINETUNING

# DEVICES

gps = robot.getDevice("gps")
gps.enable(timestep)

camera = robot.getDevice("camera(1)")
camera.enable(timestep)

leftMotor = robot.getDevice("left wheel motor")
rightMotor = robot.getDevice("right wheel motor")
leftMotor.setPosition(float("inf"))
rightMotor.setPosition(float("inf"))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

min_area = 100
max_area = 500000

max_speed = 6.28
pixel_tolerance = 50  # how centered to the object

# close enough for CNN using bounding box area
cnn_area_frac = 0.30  # stop when target box covers x% of frame
area_confirm_frames = 3  # must be above threshold this many frames in a row

height = camera.getHeight()
width = camera.getWidth()

cnn_area_stop = cnn_area_frac * (width * height)  # how many pixels should we have in frame before we stop

# stop centering jitter
uncentre_confirm_frames = 5  # must be uncentered this many frames before we turn
centre_confirm_frames = 2  # must be centred this many frames before we go forward

lost = 0
lost_frames = 5

#  A* PATHFINDING

path = None  # path needs to be remembered outside of timestep

ps0 = robot.getDevice("ps0")
ps1 = robot.getDevice("ps1")
ps2 = robot.getDevice("ps2")
ps0.enable(timestep)
ps1.enable(timestep)
ps2.enable(timestep)

arena_x = 3.0
arena_y = 2.0

nx = 60  # 60 cells wide
ny = 40  # 40 cells height

cell_x = arena_x / nx
cell_y = arena_y / ny

grid = [[0 for _ in range(ny)] for _ in range(nx)]  # initialise all cells as free until otherwise said

for ix in range(nx):
    for iy in range(ny):
        if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
            grid[ix][iy] = 1  # on the edge of the map, this is blocked space

static_occupied_cells = (
    [(41, iy) for iy in range(9, 40)])
# + [(a, b) for b in range(35, 40) for a in range(50, 55)]
# + [(a, b) for b in range(24, 29) for a in range(55, 60)]
# + [(a, b) for b in range(14, 19) for a in range(55, 60)]


for ix, iy in static_occupied_cells:
    grid[ix][iy] = 1  # extra walls and recycling points set to occupied


def clamp(v, low, high):
    # keep cells between min max cells. some objects may leak into occupied external cells
    if v < low:
        return low
    if v > high:
        return high
    return v


def gps_to_cell(x, y):
    # shift x from [-arena_x/2, +arena_x/2] into [0, arena_x]
    shifted_x = x + arena_x / 2  # map on x axis goes from -1.5 to +1.5, we are centred around 0
    shifted_y = y + arena_y / 2  # -1 to + 1 on y axis

    # convert from  meters to cells
    ix = int(shifted_x / cell_x)
    iy = int(shifted_y / cell_y)

    # clamp indices so we never go outside the indexing, lists cannot be -1 and 60 doesnt exist in lists as they count from 0
    ix = clamp(ix, 0, nx - 1)  # 60 rows in list space is 0-59
    iy = clamp(iy, 0, ny - 1)

    # return grid coordinate
    return ix, iy


def manhattan(ix, iy):  # for 4 distance movement
    # a and b are (ix, iy)
    return abs(ix[0] - iy[0]) + abs(ix[1] - iy[1])  # abs means its positive regardless of sign


def get_neighbours(cell):
    cx, cy = cell  # unpacking tuple
    candidates = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]  # all possible movements for 4 directions
    neighbours = []
    for x_candidate, y_candidate in candidates:  # check each candidate
        if 0 <= x_candidate < nx and 0 <= y_candidate < ny:  # if cells are within cell list index [0-59]
            if grid[x_candidate][y_candidate] == 0:  # and not occupied
                neighbours.append((x_candidate, y_candidate))  # they are a neighbour to check
    return neighbours


def astar(start_cell, goal_cell):
    open_heap = []
    heapq.heappush(open_heap, (
        0, start_cell))  # heapq.heappush(yourheaphere, youritemhere) my item is a tuple (f_score, start_cell)
    came_from = {}  # dictionaries for these, came from needs t know the cell it cames from
    g_score = {start_cell: 0}  # g score needs score for cell, its 0 at start
    while open_heap:
        f_score, current_cell = heapq.heappop(open_heap)  # smallest item popped off heap

        if current_cell == goal_cell:
            print("goal reached")
            path = [goal_cell]
            while path[-1] != start_cell:  # while we are not at the start
                path.append(came_from[path[-1]])  # getting the parent of the prior cells to form the path
            path.reverse()  # reverse so we can follow
            return path

        for neighbour_cell in get_neighbours(current_cell):

            # print("current:", current_cell, "neighbour:", neighbour_cell)
            tentative_g = g_score[
                              current_cell] + 1  # we can only move in one direction, tentative g for neighbour is the current cell g score + 1

            if tentative_g < g_score.get(neighbour_cell,
                                         float(
                                             "inf")):  # inf because it may be the first time we discover the cell .get(key, dafault value)
                # this finds the best g score to reach cell, is the tentative g cheaper than recorded g
                came_from[neighbour_cell] = current_cell  # update came from history for best neighbour cell
                g_score[neighbour_cell] = tentative_g  # update g score for neighbour cell

                h = manhattan(neighbour_cell, goal_cell)
                f = tentative_g + h
                heapq.heappush(open_heap, (f, neighbour_cell))
    return None


# STATE MACHINE, YOU NEED THIS THROUGH THE WHOLE PIPELINE

state = "SEARCHING"
locked_rect = None  # store outside of loop, reinitialise within loop when new target is needed
area_ok_count = 0
uncentre_count = 0
centre_count = 0


# FUNCTIONS/MATH BITS

def get_opencv_image(camera_device):
    raw = camera_device.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    img = img[:, :, :3].copy()
    return img


def visualise_mask(mask):
    cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Debug", 600, 300)
    cv2.imshow("Vision Debug", mask)
    cv2.waitKey(1)


def visualise(frame):
    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision", 600, 300)
    cv2.imshow("Vision", frame)
    cv2.waitKey(1)


def bottom_y(rect):  # looks at the bottom which helps with what appears closest to robot
    x, y, w, h = rect
    return y + h


def rect_centre(rect):
    x, y, w, h = rect
    return x + w / 2, y + h / 2


def closest_to_locked(rects, locked):
    lx, ly = rect_centre(locked)
    return min(
        rects,
        key=lambda r: (rect_centre(r)[0] - lx) ** 2 + (rect_centre(r)[1] - ly) ** 2
    )
    # finds which bounding box is the one you wanted from the last timestep,
    # its recomputed and it will forget, this finds the CLOSEST PIXEL DISTANCE TO YOUR TARGET
    # euclidean distance


# MAIN LOOP

while robot.step(timestep) != -1:  # remember its one big loop, the robot has no memory so you need to give it memory
    leftSpeed = 0.0  # resets movemnent per timestep
    rightSpeed = 0.0

    # SENSING

    frame = get_opencv_image(camera)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # remove speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps
    visualise_mask(mask)

    # Find contours and bounding rects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:  # filtering object noise
            valid_contours.append(cnt)

    bounding_rects = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # if any box includes an edge skip it move to next contour
        if x == 0 or y == 0:  # this might cause getting 'lost' when object bounding box gets closer and y and x become on edge
            continue
        bounding_rects.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    visualise(frame)

    # distance sensor debug not using anymore but nice to have
    val0 = ps0.getValue()
    val1 = ps1.getValue()
    val2 = ps2.getValue()
    front_aprox = (val0 + val1 + val2) / 3
    # print("front aprox:", front_aprox, "| state:", state)

    # STATE: SEARCHING

    if state == "SEARCHING":
        if len(bounding_rects) > 0:
            # Lock immediately on the closest object
            locked_rect = max(bounding_rects, key=bottom_y)
            area_ok_count = 0
            uncenter_count = 0
            center_count = 0
            lost = 0
            state = "APPROACHING"
            print("LOCKED  APPROACHING")
        else:
            # Keep scanning
            leftSpeed = 0.2 * max_speed
            rightSpeed = -0.2 * max_speed
            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            continue

    # STATE: APPROACHING

    if state == "APPROACHING":
        # if we have detections this frame, update where the locked target is
        if len(bounding_rects) > 0 and locked_rect is not None:
            target_rect = closest_to_locked(bounding_rects, locked_rect)
            locked_rect = target_rect
            lost = 0
        else:
            # if we flicker dont pick a new target straightaway
            lost += 1
            if lost < lost_frames:
                print("lost for", lost)
                # wait to spin
                leftSpeed = 0.0 * max_speed
                rightSpeed = 0.0 * max_speed
                leftMotor.setVelocity(leftSpeed)
                rightMotor.setVelocity(rightSpeed)
                continue

            # spin until you find again or set as lost and retunr to searching

            leftSpeed = 0.15 * max_speed
            rightSpeed = -0.15 * max_speed
            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            uncenter_count = 0
            center_count = 0
            area_ok_count = 0
            continue

        # Use the locked rect for steering
        x, y, w, h = locked_rect
        cx = x + w / 2
        centre_x = width / 2
        error_x = cx - centre_x
        # print(
        # f"RECT: x={x}, y={y}, w={w}, h={h}, "
        # f"cx={cx:.1f}, error_x={error_x:.1f}, area={w*h}, n={len(bounding_rects)}")

        # debounce centring logic
        off_centre = abs(error_x) > pixel_tolerance

        if off_centre:
            uncentre_count += 1
            centre_count = 0
        else:
            centre_count += 1
            uncentre_count = 0

        # only turn if uncentred for multiple frames stops it panicking at one flicker
        if uncentre_count >= uncentre_confirm_frames:
            print(f"TURN: error_x={error_x:.1f}, cx={cx:.1f}, num of boxes={len(bounding_rects)}")

            # Dont count image readiness while turning
            area_ok_count = 0

            if error_x > 0:
                leftSpeed = 0.2 * max_speed
                rightSpeed = -0.2 * max_speed
                print("turning right to centre")
            else:
                leftSpeed = -0.2 * max_speed
                rightSpeed = 0.2 * max_speed
                print("turning left to centre")

        elif centre_count >= centre_confirm_frames:
            # decide if close enough for CNN using bounding box area
            box_area = w * h
            print(f"box area={box_area:.0f} threshold={cnn_area_stop:.0f}")

            if box_area >= cnn_area_stop:
                area_ok_count += 1
            else:
                area_ok_count = 0

            if area_ok_count >= area_confirm_frames:
                leftSpeed = 0.0
                rightSpeed = 0.0
                print("Close enough, let's get a picture")
                state = "PATHFIND"

            else:
                # Approach forward
                leftSpeed = 0.2 * max_speed
                rightSpeed = 0.2 * max_speed
                print("centred, going forward")

        leftMotor.setVelocity(leftSpeed)
        rightMotor.setVelocity(rightSpeed)

    # STATE: CNN CAPTURE

    if state == "CNN_CAPTURE":
        # print("CNN_CAPTURE READY")
        state = "PATHFIND"
        # if cnn = recycle point identifier:

    if state == "PATHFIND":
        recycle_coord = (1.11, 0.89)
        goal_world_x, goal_world_y = recycle_coord
        goal_cell = gps_to_cell(goal_world_x, goal_world_y)
        print("Goal cell:", goal_cell)

        position = gps.getValues()  # need to know where you are to move into cell space
        x = position[0]
        y = position[1]

        robot_cell = gps_to_cell(x, y)
        # print(f"gps: ({x:.2f},{y:.2f}) cell:",robot_cell)
        path = astar(robot_cell, goal_cell)
        print(path)

    # astar(grid, start_cell, goal_cell)
    # edt astar for obstracle detection and replanning
    # set motors

    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
