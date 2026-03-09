from controller import Robot
import numpy as np
import cv2
import heapq
import math


class RobotDevices:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # DEVICES

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        self.inertial_unit = self.robot.getDevice("inertial unit")
        self.inertial_unit.enable(self.timestep)

        self.camera = self.robot.getDevice("camera(1)")
        self.camera.enable(self.timestep)

        self.leftMotor = self.robot.getDevice("left wheel motor")
        self.rightMotor = self.robot.getDevice("right wheel motor")
        self.leftMotor.setPosition(float("inf"))
        self.rightMotor.setPosition(float("inf"))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.ps0 = self.robot.getDevice("ps0")
        self.ps1 = self.robot.getDevice("ps1")
        self.ps2 = self.robot.getDevice("ps2")
        self.ps0.enable(self.timestep)
        self.ps1.enable(self.timestep)
        self.ps2.enable(self.timestep)

        self.height = self.camera.getHeight()
        self.width = self.camera.getWidth()

    def step(self):
        return self.robot.step(self.timestep)

    def set_wheel_speeds(self, leftSpeed, rightSpeed):
        self.leftMotor.setVelocity(leftSpeed)
        self.rightMotor.setVelocity(rightSpeed)

    def get_pose(self):
        position = self.gps.getValues()  # need to know where you are to move into cell space
        x = position[0]
        y = position[1]
        robot_yaw = self.inertial_unit.getRollPitchYaw()[2]
        return x, y, robot_yaw

    def get_front_approx(self):  # fixed missing colon
        val0 = self.ps0.getValue()
        val1 = self.ps1.getValue()
        val2 = self.ps2.getValue()
        return (val0 + val1 + val2) / 3

    def get_opencv_image(self):
        raw = self.camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 4))
        img = img[:, :, :3].copy()
        return img


class GridMap:
    def __init__(self):
        self.arena_x = 3.0
        self.arena_y = 2.0

        self.nx = 60  # 60 cells wide
        self.ny = 40  # 40 cells height

        self.cell_x = self.arena_x / self.nx
        self.cell_y = self.arena_y / self.ny

        self.grid = [[0 for _ in range(self.ny)] for _ in range(self.nx)]  # initialise all cells as free until otherwise said

        for ix in range(self.nx):
            for iy in range(self.ny):
                if ix == 0 or ix == self.nx - 1 or iy == 0 or iy == self.ny - 1:
                    self.grid[ix][iy] = 1  # on the edge of the map, this is blocked space

        self.static_occupied_cells = (
            [(41, iy) for iy in range(9, 40)]
            + [(40, iy) for iy in range(9, 40)]
            + [(42, iy) for iy in range(9, 40)]
            + [(41, 8), (40, 8), (42, 8)]
        )

        self.dynamic_occupied_cells = []  # insert obstacles found at runtime

        for ix, iy in self.static_occupied_cells:
            self.grid[ix][iy] = 1  # extra walls and recycling points set to occupied

    def mark_obstacle(self, ix, iy):
        self.grid[ix][iy] = 1
        self.dynamic_occupied_cells.append((ix, iy))  # track separately from static

    def clear_dynamic_obstacles(self):
        for ix, iy in self.dynamic_occupied_cells:
            self.grid[ix][iy] = 0
        self.dynamic_occupied_cells = [] # clear if no path can be found

    def clamp(self, v, low, high):
        # keep cells between min max cells. some objects may leak into occupied external cells
        if v < low:
            return low
        if v > high:
            return high
        return v

    def gps_to_cell(self, x, y):
        # shift x from [-arena_x/2, +arena_x/2] into [0, arena_x]
        shifted_x = x + self.arena_x / 2  # map on x axis goes from -1.5 to +1.5, we are centred around 0
        shifted_y = y + self.arena_y / 2  # -1 to + 1 on y axis

        # convert from  meters to cells
        ix = int(shifted_x / self.cell_x)
        iy = int(shifted_y / self.cell_y)

        # clamp indices so we never go outside the indexing, lists cannot be -1 and 60 doesnt exist in lists as they count from 0
        ix = self.clamp(ix, 0, self.nx - 1)  # 60 rows in list space is 0-59
        iy = self.clamp(iy, 0, self.ny - 1)

        # return grid coordinate
        return ix, iy

    def cell_to_gps(self, ix, iy):
        left_edge_arena = -self.arena_x / 2.0  # arena index starts from bottom left edge
        bottom_edge_arena = -self.arena_y / 2.0  # -1.5, -1 is index [0,0]

        target_world_x = left_edge_arena + (
            ix + 0.5
        ) * self.cell_x  # start from bottom left, add ixy + half cell length to get centre of cell
        target_world_y = bottom_edge_arena + (
            iy + 0.5
        ) * self.cell_y  # then * by cell physical size to get GPS location

        return target_world_x, target_world_y

    def manhattan(self, ix, iy):  # for 4 distance movement
        # a and b are (ix, iy)
        return abs(ix[0] - iy[0]) + abs(ix[1] - iy[1])  # abs means its positive regardless of sign

    def get_neighbours(self, cell):
        cx, cy = cell  # unpacking tuple
        candidates = [
            (cx + 1, cy),      # right
            (cx - 1, cy),      # left
            (cx, cy + 1),      # up
            (cx, cy - 1),      # down
            (cx + 1, cy + 1),  # diagonal up right
            (cx + 1, cy - 1),  # diagonal down right
            (cx - 1, cy + 1),  # diagonal up left
            (cx - 1, cy - 1),  # diagonal down left
        ]  # all possible movements for 8 directions

        neighbours = []
        for x_candidate, y_candidate in candidates:  # check each candidate
            if 0 <= x_candidate < self.nx and 0 <= y_candidate < self.ny:  # if cells are within cell list index [0-59]
                if self.grid[x_candidate][y_candidate] == 0:  # and not occupied
                    neighbours.append((x_candidate, y_candidate))  # they are a neighbour to check
        return neighbours


class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map

    def euclidean(self, neighbour_cell, goal_cell):
        dx = goal_cell[0] - neighbour_cell[0]
        dy = goal_cell[1] - neighbour_cell[1]
        return math.sqrt(dx * dx + dy * dy)

    def astar(self, start_cell, goal_cell):
        open_heap = []
        heapq.heappush(open_heap, (
            0, start_cell))  # heapq.heappush(yourheaphere, youritemhere) my item is a tuple (f_score, start_cell)

        came_from = {}  # dictionaries for these, came from needs to know the cell it came from
        g_score = {start_cell: 0}  # g score needs score for cell, its 0 at start

        while open_heap:
            f_score, current_cell = heapq.heappop(open_heap)  # smallest item popped off heap

            if current_cell == goal_cell:
                path = [goal_cell]
                while path[-1] != start_cell:  # while we are not at the start
                    path.append(came_from[path[-1]])  # getting the parent of the prior cells to form the path
                path.reverse()  # reverse so we can follow
                return path

            for neighbour_cell in self.grid_map.get_neighbours(current_cell):

                dx = neighbour_cell[0] - current_cell[0]  # check if we moved diagonally
                dy = neighbour_cell[1] - current_cell[1]  # if these are both not 0, we did
                step_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0  # if we moved diagonally step cost is square 2
                tentative_g = g_score[current_cell] + step_cost  # tentative g for neighbour is the current cell g score + movement cost

                if tentative_g < g_score.get(
                    neighbour_cell,
                    float("inf")
                ):  # inf because it may be the first time we discover the cell .get(key, default value)
                    came_from[neighbour_cell] = current_cell  # update came from history for best neighbour cell
                    g_score[neighbour_cell] = tentative_g  # update g score for neighbour cell

                    h = self.euclidean(neighbour_cell, goal_cell)
                    f = tentative_g + h
                    heapq.heappush(open_heap, (f, neighbour_cell))

        return None


class Navigator:
    def __init__(self, devices, max_speed):
        self.devices = devices
        self.max_speed = max_speed
        self.angle_close_enough = 0.20
        self.waypoint_reached_distance = 0.03

    def drive_to_waypoint(self, target_world_x, target_world_y):
        x, y, robot_yaw = self.devices.get_pose()

        vector_x = target_world_x - x  # how much we need to vector to target
        vector_y = target_world_y - y  # target subtract current location

        distance_to_target = math.sqrt(vector_x ** 2 + vector_y ** 2)  # euclidean distance

        if distance_to_target < self.waypoint_reached_distance:
            return True, 0.0, 0.0  # at target

        yaw_needed = math.atan2(vector_y, vector_x)
        yaw_to_rotate = yaw_needed - robot_yaw

        while yaw_to_rotate > math.pi:
            yaw_to_rotate -= 2 * math.pi
        while yaw_to_rotate < -math.pi:
            yaw_to_rotate += 2 * math.pi

        if abs(yaw_to_rotate) > self.angle_close_enough:  # turn to waypoint

            if yaw_to_rotate > 0:  # turn right
                leftSpeed = -(0.3 * self.max_speed)
                rightSpeed = 0.3 * self.max_speed
            else:  # turn left
                leftSpeed = 0.3 * self.max_speed
                rightSpeed = -(0.3 * self.max_speed)

            return False, leftSpeed, rightSpeed  # not at target

        else:  # forward to waypoint
            leftSpeed = 0.5 * self.max_speed
            rightSpeed = 0.5 * self.max_speed
            return False, leftSpeed, rightSpeed  # not at target


class VisionSystem:
    def __init__(self, devices, max_speed):
        self.devices = devices
        self.max_speed = max_speed

        # CONSTANTS / FINETUNING
        self.min_area = 100
        self.max_area = 500000
        self.pixel_tolerance = 50  # how centered to the object

        # close enough for CNN using bounding box area
        self.cnn_area_frac = 0.30  # stop when target box covers x% of frame
        self.area_confirm_frames = 3  # must be above threshold this many frames in a row

        self.cnn_area_stop = self.cnn_area_frac * (
            self.devices.width * self.devices.height
        )  # how many pixels should we have in frame before we stop

        # stop centering jitter
        self.uncentre_confirm_frames = 5  # must be uncentered this many frames before we turn
        self.centre_confirm_frames = 2  # must be centred this many frames before we go forward

        self.lost = 0
        self.lost_frames = 5

        # STATE MEMORY, YOU NEED THIS OUTSIDE THE TIMESTEP
        self.locked_rect = None  # store outside of loop, reinitialise within loop when new target is needed
        self.area_ok_count = 0
        self.uncentre_count = 0
        self.centre_count = 0

    def reset_tracking_counts(self):
        self.area_ok_count = 0
        self.uncentre_count = 0
        self.centre_count = 0

    def reset_all_tracking(self):
        self.locked_rect = None
        self.lost = 0
        self.reset_tracking_counts()

    def visualise_mask(self, mask):
        cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision Debug", 600, 300)
        cv2.imshow("Vision Debug", mask)
        cv2.waitKey(1)

    def visualise(self, frame):
        cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision", 600, 300)
        cv2.imshow("Vision", frame)
        cv2.waitKey(1)

    def bottom_y(self, rect):  # looks at the bottom which helps with what appears closest to robot
        x, y, w, h = rect
        return y + h

    def rect_centre(self, rect):
        x, y, w, h = rect
        return x + w / 2, y + h / 2

    def closest_to_locked(self, rects, locked):
        lx, ly = self.rect_centre(locked)
        return min(
            rects,
            key=lambda r: (self.rect_centre(r)[0] - lx) ** 2 + (self.rect_centre(r)[1] - ly) ** 2
        )

    def sense_objects(self):
        # SENSING

        frame = self.devices.get_opencv_image()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # remove speckle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps
        self.visualise_mask(mask)

        # Find contours and bounding rects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:  # filtering object noise
                valid_contours.append(cnt)

        bounding_rects = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # if any box includes an edge skip it move to next contour
            if x == 0 or y == 0:  # this might cause getting 'lost' when object bounding box gets closer and y and x become on edge
                continue
            bounding_rects.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.visualise(frame)
        return bounding_rects

    def handle_searching(self):
        bounding_rects = self.sense_objects()

        if len(bounding_rects) > 0:
            # Lock immediately on the closest object
            self.locked_rect = max(bounding_rects, key=self.bottom_y)
            self.reset_tracking_counts()
            self.lost = 0
            print("LOCKED APPROACHING")
            return "APPROACHING", 0.0, 0.0
        else:
            # Keep scanning
            leftSpeed = 0.2 * self.max_speed
            rightSpeed = -0.2 * self.max_speed
            return "SEARCHING", leftSpeed, rightSpeed

    def handle_approaching(self):
        bounding_rects = self.sense_objects()

        # if we have detections this frame, update where the locked target is
        if len(bounding_rects) > 0 and self.locked_rect is not None:
            target_rect = self.closest_to_locked(bounding_rects, self.locked_rect)
            self.locked_rect = target_rect
            self.lost = 0
        else:
            # if we flicker dont pick a new target straightaway
            self.lost += 1
            if self.lost < self.lost_frames:
                print("lost for", self.lost)
                # wait to spin
                leftSpeed = 0.0 * self.max_speed
                rightSpeed = 0.0 * self.max_speed
                return "APPROACHING", leftSpeed, rightSpeed

            # lost is greater than lost frame limit
            # spin until you find again or set as lost and return to searching
            leftSpeed = 0.15 * self.max_speed
            rightSpeed = -0.15 * self.max_speed
            self.reset_tracking_counts()

            if self.lost > 10:
                state = "SEARCHING"  # youre mega lost give up and search again
                self.locked_rect = None  # give up on that rect
                leftSpeed = 0.0 * self.max_speed
                rightSpeed = 0.0 * self.max_speed
                return state, leftSpeed, rightSpeed  # stop spinning to return to search

            return "APPROACHING", leftSpeed, rightSpeed

        # Use the locked rect for steering
        x, y, w, h = self.locked_rect
        cx = x + w / 2
        centre_x = self.devices.width / 2
        error_x = cx - centre_x

        # debounce centring logic
        off_centre = abs(error_x) > self.pixel_tolerance

        if off_centre:
            self.uncentre_count += 1
            self.centre_count = 0
        else:
            self.centre_count += 1
            self.uncentre_count = 0

        # only turn if uncentred for multiple frames stops it panicking at one flicker
        if self.uncentre_count >= self.uncentre_confirm_frames:
            print(f"TURN: error_x={error_x:.1f}, cx={cx:.1f}, num of boxes={len(bounding_rects)}")

            # Dont count image readiness while turning
            self.area_ok_count = 0

            if error_x > 0:
                leftSpeed = 0.2 * self.max_speed
                rightSpeed = -0.2 * self.max_speed
                print("turning right to centre")
            else:
                leftSpeed = -0.2 * self.max_speed
                rightSpeed = 0.2 * self.max_speed
                print("turning left to centre")

            return "APPROACHING", leftSpeed, rightSpeed

        elif self.centre_count >= self.centre_confirm_frames:
            # decide if close enough for CNN using bounding box area
            box_area = w * h

            if box_area >= self.cnn_area_stop:
                self.area_ok_count += 1
            else:
                self.area_ok_count = 0

            if self.area_ok_count >= self.area_confirm_frames:
                leftSpeed = 0.0
                rightSpeed = 0.0
                print("Close enough, let's get a picture")
                state = "PATHFIND"
                return state, leftSpeed, rightSpeed

            else:
                # Approach forward
                leftSpeed = 0.2 * self.max_speed
                rightSpeed = 0.2 * self.max_speed
                return "APPROACHING", leftSpeed, rightSpeed

        else:
            leftSpeed = 0.0  # mitigate stale speed if centre counts not met
            rightSpeed = 0.0
            return "APPROACHING", leftSpeed, rightSpeed


class RobotController:
    def __init__(self):
        self.devices = RobotDevices()

        self.max_speed = 6.28

        self.grid_map = GridMap()
        self.planner = AStarPlanner(self.grid_map)
        self.navigator = Navigator(self.devices, self.max_speed)
        self.vision = VisionSystem(self.devices, self.max_speed)

        # STATE MACHINE, YOU NEED THIS THROUGH THE WHOLE PIPELINE
        self.state = "SEARCHING"

        # A* PATHFINDING MEMORY
        self.planned_path = None  # path needs to be remembered outside timestep
        self.current_path_cell = 0
        self.travelling = "recycle_point"

        self.metal_recycle_coord = (1.11, 0.89)
        self.world_reset = (0, 0)

        # tune this for threshold
        self.obstacle_threshold = 50

    def check_for_obstacles(self):
        front_distance = self.devices.get_front_approx()

        if front_distance > self.obstacle_threshold:
            x, y, robot_yaw = self.devices.get_pose()

            # estimate cell directly in front based on yaw
            obstacle_x = x + math.cos(robot_yaw) * self.grid_map.cell_x
            obstacle_y = y + math.sin(robot_yaw) * self.grid_map.cell_y

            ix, iy = self.grid_map.gps_to_cell(obstacle_x, obstacle_y)
            self.grid_map.mark_obstacle(ix, iy)  # mark in grid and dynamic list
            self.planned_path = None  # wipe path
            print(f"Obstacle detected at cell ({ix}, {iy}), replanning")

    def handle_cnn_capture(self):
        return "PATHFIND", 0.0, 0.0

    def handle_pathfind(self):

        self.check_for_obstacles()
        goal_cell = None # incase never assigned
        if self.travelling == "home":
            goal_world_x, goal_world_y = self.world_reset
            goal_cell = self.grid_map.gps_to_cell(goal_world_x, goal_world_y)

        elif self.travelling == "recycle_point":
            goal_world_x, goal_world_y = self.metal_recycle_coord
            goal_cell = self.grid_map.gps_to_cell(goal_world_x, goal_world_y)

        x, y, robot_yaw = self.devices.get_pose()
        robot_cell = self.grid_map.gps_to_cell(x, y)

        if self.planned_path is None:
            self.planned_path = self.planner.astar(robot_cell, goal_cell)
            self.current_path_cell = 0
            print("replanning from", robot_cell, "to", goal_cell)

            if self.planned_path is None:  # if a* cant find a path
                print("No safe path found")
                return "PATHFIND", 0.0, 0.0

        if self.current_path_cell >= len(self.planned_path):  # if we have finished the path
            self.planned_path = None      # wipe old path so A* replans for next
            self.current_path_cell = 0    # reset index

            if self.travelling == "recycle_point":
                self.travelling = "home"  # if we are at the recycle point, change to home to go to after
                return "PATHFIND", 0.0, 0.0

            elif self.travelling == "home":  # if we are at home (0,0)
                self.travelling = "recycle_point"  # reintialise to recycle point for next object
                self.vision.reset_all_tracking()
                return "SEARCHING", 0.0, 0.0  # back to looking for objects

        target_cell = self.planned_path[self.current_path_cell]

        target_cell_x = target_cell[0]
        target_cell_y = target_cell[1]

        target_world_x, target_world_y = self.grid_map.cell_to_gps(target_cell_x, target_cell_y)

        reached, leftSpeed, rightSpeed = self.navigator.drive_to_waypoint(target_world_x, target_world_y)

        if reached is True:
            self.current_path_cell += 1  # increment after each waypoint is reached

        return "PATHFIND", leftSpeed, rightSpeed

    def run(self):
        # MAIN LOOP
        while self.devices.step() != -1:
            leftSpeed = 0.0  # resets movement per timestep
            rightSpeed = 0.0

            # STATE: SEARCHING
            if self.state == "SEARCHING":
                self.state, leftSpeed, rightSpeed = self.vision.handle_searching()

            # STATE: APPROACHING
            elif self.state == "APPROACHING":
                self.state, leftSpeed, rightSpeed = self.vision.handle_approaching()

            # STATE: CNN CAPTURE
            elif self.state == "CNN_CAPTURE":
                self.state, leftSpeed, rightSpeed = self.handle_cnn_capture()

            # STATE: PATHFIND
            elif self.state == "PATHFIND":
                self.state, leftSpeed, rightSpeed = self.handle_pathfind()

            self.devices.set_wheel_speeds(leftSpeed, rightSpeed)


if __name__ == "__main__":
    controller = RobotController()
    controller.run()