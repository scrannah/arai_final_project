from controller import Supervisor

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

display = supervisor.getDevice("grid")
W = display.getWidth()
H = display.getHeight()

NX, NZ = 60, 40  # grid size

static_occupied_cells = (
    [(41, b) for b in range(9, 40)] +
    [(a, b) for b in range(35, 40) for a in range(50, 55)] +
    [(a, b) for b in range(24, 29) for a in range(55, 60)] +
    [(a, b) for b in range(14, 19) for a in range(55, 60)]
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

    # draw grid lines
    display.setColor(0x444444)

    for a in range(NX + 1):
        x = int(a * W / NX)
        display.drawLine(x, 0, x, H)

    for b in range(NZ + 1):
        y = int(b * H / NZ)
        display.drawLine(0, y, W, y)
