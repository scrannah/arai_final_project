from controller import Robot
import numpy as np
import cv2

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# CONSTANTS / FINETUNING

# DEVICES

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
cnn_area_frac = 0.35  # stop when target box covers x% of frame
area_confirm_frames = 3  # must be above threshold this many frames in a row

height = camera.getHeight()
width = camera.getWidth()

cnn_area_stop = cnn_area_frac * (width * height)  # how many pixels should we have in frame before we stop

# stop centering jitter
uncentre_confirm_frames = 5  # must be uncentred this many frames before we turn
centre_confirm_frames = 2  # must be centred this many frames before we go forward

lost = 0
lost_frames = 5

# PATHFINDING

ps0 = robot.getDevice("ps0")
ps1 = robot.getDevice("ps1")
ps2 = robot.getDevice("ps2")
ps0.enable(timestep)
ps1.enable(timestep)
ps2.enable(timestep)

# PLAN ARENA OUT FOR A*

arena_x = 3.0
arena_y = 2.0

nx = 60
ny = 40

cell_x = arena_x / nx
cell_y = arena_y / ny

grid = [[0 for a in range()] for b in range(nx)]  # initialise all cells as free until otherwise said

for a in range(nx):
    for b in range(ny):
        if a == 0 or a == nx - 1 or b == 0 or b == ny - 1:
            grid[a][b] = 1  # youre on the edge of the map, this is blocked space

static_occupied_cells = (
        [(41, b) for b in range(9, 40)] +
        [(a, b) for b in range(35, 40) for a in range(50, 55)] +
        [(a, b) for b in range(24, 29) for a in range(55, 60)] +
        [(a, b) for b in range(14, 19) for a in range(55, 60)]
)

for a, b in static_occupied_cells:
    grid[a][b] = 1  # extra walls and recycling points set to occupied

# STATE MACHINE, YOU NEED THIS THROUGH THE WHOLE PIPELINE

state = "SEARCHING"
locked_rect = None  # store outside of loop, reinitialise within loop when new target is needed
area_ok_count = 0
uncenter_count = 0
centre_count = 0


# FUNCTIONS/MATHY BITS

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


def bottom_y(rect):
    x, y, w, h = rect
    return y + h


def rect_centre(rect):
    x, y, w, h = rect
    return (x + w / 2, y + h / 2)


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
                state = "CNN_CAPTURE"
                print("Close enough, let's get a picture")
                state = 'SEARCHING'
            else:
                # Approach forward
                leftSpeed = 0.2 * max_speed
                rightSpeed = 0.2 * max_speed
                print("centred, going forward")

        leftMotor.setVelocity(leftSpeed)
        rightMotor.setVelocity(rightSpeed)

    # STATE: CNN CAPTURE

    if state == "CNN_CAPTURE":
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        print("CNN_CAPTURE READY")
        break
        # get picture
        # pathfind
        # reinitialise locked rect

    # if state == "PATHFIND":
    # PATHFIND

    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
