# This can not be imported, lel
if __name__ != "__main__":
    quit()

import sys
import argparse

import numpy as np
from matplotlib import cm

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import lbm

def get_cmap_from_matplotlib(cmap=cm.coolwarm):
    # Create colormap for OpenGL plotting
    ncol = cmap.N
    cmap_rgba = []
    for i in range(ncol - 1):
        b, g, r, _ = cmap(
            i)  # Not sure why this is inverted, I was expecting r, g, b.
        cmap_rgba.append(
            int(255.0) << 24 | (int(float(r) * 255.0) << 16) | (
                int(float(g) * 255.0) << 8) | (int(float(b) * 255.0) << 0))
    return np.array(cmap_rgba), len(cmap_rgba)

def display():
    plotvar = get_var_map[show]()
    minvar = 0
    maxvar = 1.001 * (np.max(plotvar))

    # convert the plotvar array into an array of colors to plot
    # if the mesh point is solid, make it white
    frac = (plotvar[:] - minvar) / (maxvar - minvar)
    icol = frac * ncol
    plot_rgba[:] = cmap_rgba[icol.astype(np.int)]
    plot_rgba[model.obstacle.T.flatten()] = 0xFF000000  #Color code of white

    # Fill the pixel buffer with the plot_rgba array
    glBufferData(GL_PIXEL_UNPACK_BUFFER, plot_rgba.nbytes, plot_rgba,
                 GL_STREAM_COPY)

    # Copy the pixel buffer to the texture, ready to display
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_UNSIGNED_BYTE,
                    None)

    # Render one quad to the screen and colour it using our texture
    # i.e. plot our plotvar data to the screen
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_QUADS)

    x0, y0 = 0.0, 0.0
    x1, y1 = nx, ny

    glTexCoord2f(0.0, 0.0)
    glVertex3f(x0, y0, 0.0)

    glTexCoord2f(1.0, 0.0)
    glVertex3f(x1, y0, 0.0)

    glTexCoord2f(1.0, 1.0)
    glVertex3f(x1, y1, 0.0)

    glTexCoord2f(0.0, 1.0)
    glVertex3f(x0, y1, 0.0)

    glEnd()
    glutSwapBuffers()


def resize(w, h):
    #GLUT resize callback to allow us to change the window size.
    global width, height
    width = w
    height = h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0., nx, 0., ny, -200., 200.)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def mouse(button, state, x, y):
    global draw_solid_flag, ipos_old, jpos_old
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        draw_solid_flag = 0
        xx = x
        yy = y
        ipos_old = int(float(xx) / width * float(nx))
        jpos_old = int(float(height - yy) / height * float(ny))

    if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
        draw_solid_flag = 1
        xx = x
        yy = y
        ipos_old = int(float(xx) / width * float(nx))
        jpos_old = int(float(height - yy) / height * float(ny))


def mouse_motion(x, y):
    '''
    GLUT call back for when the mouse is moving
    This sets the solid array to draw_solid_flag as set in the mouse callback
    It will draw a staircase line if we move more than one pixel since the
    last callback - that makes the coding a bit cumbersome:
    '''
    global ipos_old, jpos_old

    xx = x
    yy = y
    ipos = int(float(xx) / width * float(nx))
    jpos = int(float(height - yy) / height * float(ny))

    if ipos <= ipos_old:
        i1 = ipos
        i2 = ipos_old
        j1 = jpos
        j2 = jpos_old

    else:
        i1 = ipos_old
        i2 = ipos
        j1 = jpos_old
        j2 = jpos

    jlast = j1

    for i in range(i1, i2 + 1):
        if i1 != i2:
            frac = (i - i1) / (i2 - i1)
            jnext = int((frac * (j2 - j1)) + j1)
        else:
            jnext = j2

        if jnext >= jlast:
            model.obstacle[i, jlast] = draw_solid_flag

            for j in range(jlast, jnext + 1):
                model.obstacle[i, j] = draw_solid_flag
        else:
            solid[i, jlast] = draw_solid_flag
            for j in range(jnext, jlast + 1):
                model.obstacle[i, j] = draw_solid_flag

        jlast = jnext

    ipos_old = ipos
    jpos_old = jpos


def idle():
    global frameCount, previousTime

    frameCount = frameCount + 1.0

    currentTime = glutGet(GLUT_ELAPSED_TIME)

    timeInterval = currentTime - previousTime

    # Take an LBM step
    if not paused:
        model.step()

    if (timeInterval > 1000):
        fps = frameCount / (timeInterval / 1000.0)
        previousTime = currentTime
        frameCount = 0.0
        glutSetWindowTitle(name + " - " + str(fps) + "FPS")    

    glutPostRedisplay()

def toggle_pause():
    global paused
    paused = not paused

def set_show(val):
    global show
    show = val

def set_showi(val):
    global showi
    showi = val

def keyboard(*args):
    try:
        key_action_map[args[0]]()
    except KeyError:
        pass
    
def run_opengl():
    # OpenGL setup
    glutInit(name)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(nx * 1, ny * 1)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(name)

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, nx, 0., ny, -200.0, 200.0)

    glEnable(GL_TEXTURE_2D)

    gl_Tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, gl_Tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, None)

    gl_PBO = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO)

    #setup callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(resize)
    glutIdleFunc(idle)
    glutMouseFunc(mouse)
    glutMotionFunc(mouse_motion)
    glutKeyboardFunc(keyboard)

    # Start main loop
    glutMainLoop()


# Constants, Globals

name = 'Aerodynamics Test'
frameCount, previousTime = 0.0, 0.0  # FPS counter vars
show = 'velocity'
showi = 0
paused = 'False'
plot_rgba = None
model = None
nx, ny = None, None

get_var_map = {
    'velocity': lambda: model.velocity.T.flatten(),
    'density': lambda: model.density.T.flatten(),
    'population': lambda: model.population[showi].T.flatten(),
    'equilibrium': lambda: model.equilibrium[showi].T.flatten()
}

key_action_map = {
    b'p': toggle_pause,
    b'r': lambda: model.reset(),
    b'd': lambda: set_show('density'),
    b'v': lambda: set_show('velocity'),
    b'f': lambda: set_show('population'),
    b'e': lambda: set_show('equilibrium'),
    b'q': lambda: sys.exit(),
    b'0': lambda: set_showi(0),
    b'1': lambda: set_showi(1),
    b'2': lambda: set_showi(2),
    b'3': lambda: set_showi(3),
    b'4': lambda: set_showi(4),
    b'5': lambda: set_showi(5),
    b'6': lambda: set_showi(6),
    b'7': lambda: set_showi(7),
    b'8': lambda: set_showi(8),
    b'\x1b': lambda: sys.exit()
}

# Main
parser = argparse.ArgumentParser()
parser.add_argument("nx", help='width of lattice grid', type=int)
parser.add_argument("ny", help='height of lattice grid', type=int)
parser.add_argument("shape", help='shape of object', type=str, choices=['square', 'circle', 'wall', 'none'])
args = parser.parse_args()

model = lbm.Model(shape=(args.nx, args.ny), obstacle=args.shape)
nx, ny = model.shape
plot_rgba = np.zeros(np.prod(model.shape), dtype=np.uint32)
cmap_rgba, ncol = get_cmap_from_matplotlib()

for i in range(1):
    model.step()

run_opengl()
