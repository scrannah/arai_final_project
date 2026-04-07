from controller import Supervisor
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

receiver = supervisor.getDevice("receiver")
receiver.enable(timestep)

display = supervisor.getDevice("grid")
W = display.getWidth()
H = display.getHeight()

NX, NZ = 60, 40  # grid size

static_occupied_cells = (
        [(ix, iy) for ix in range(39, 45) for iy in range(11, 40)]  # long wall + buffer
        +
        [(ix, iy) for ix in range(55, 60) for iy in range(28, 33)]  # wood obstacle
        +
        [(ix, iy) for ix in range(55, 60) for iy in range(21, 26)]  # cardboard dropoff
        +
        [(ix, iy) for ix in range(50, 55) for iy in range(35, 40)]  # metal dropoff
)

while supervisor.step(timestep) != -1:
    # clear background

    display.setColor(0x000000)
    display.fillRectangle(0, 0, W, H)

    cell_w = W / NX
    cell_h = H / NZ

    # draw occupied cells first (so grid lines are drawn on top)
    display.setColor(0xFF0000)  # red
    for (a, b) in static_occupied_cells:
        x = int(a * cell_w)
        y = int((NZ - 1 - b) * cell_h)
        display.fillRectangle(x, y, int(cell_w), int(cell_h))

    # display.setColor(0x95C8D8)  # blue
    # for (a, b) in dynamic_occupied_cells:
    # x = int(a * cell_w)
    # y = int((NZ - 1 - b) * cell_h)
    # display.fillRectangle(x, y, int(cell_w), int(cell_h))
    # draw grid lines
    display.setColor(0x444444)

    for a in range(NX + 1):
        x = int(a * W / NX)
        display.drawLine(x, 0, x, H)

    for b in range(NZ + 1):
        y = int(b * H / NZ)
        display.drawLine(0, y, W, y)

    if receiver.getQueueLength() > 0:
        x, y = receiver.getString().split(",")
        robot_x, robot_y = float(x), float(y)
        while receiver.getQueueLength() > 0:
            receiver.nextPacket()
        print("coord received")

        objects_group = supervisor.getFromDef("Objects")
        children_field = objects_group.getField("children")
        count = children_field.getCount()
        print("object count:", count)

        closest_node = None  # reset each time
        closest_dist = float("inf")  # reset each time

        for i in range(count):
            child = children_field.getMFNode(i)
            obj_x, obj_y, _ = child.getField("translation").getSFVec3f()
            dist = math.sqrt((obj_x - robot_x) ** 2 + (obj_y - robot_y) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_node = child

        if closest_node is not None:
            closest_node.remove()