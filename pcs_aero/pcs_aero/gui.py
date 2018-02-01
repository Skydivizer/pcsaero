# -*- coding: utf-8 -*-
"""This module defines the graphical display of the models using opengl

This file is heaviliy based on the sample provided by Dr Gabor.
"""

import sys

import numpy as np
from matplotlib import cm

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *


# Constants, Globals
name = 'Drag'  # Window name
frameCount, previousTime = 0.0, 0.0  # FPS counter vars
show = 'velocity'  # what variable to show
showi = 0  # what index to show in multi dim variables
paused = False
plot_rgba = None
model = None  # global pointer to running model
nx, ny = None, None

# show to variable mapping
get_var_map = {
    'velocity': lambda: model.velocity.T.flatten(),
    'density': lambda: model.density.T.flatten(),
    'population': lambda: model.population[showi % 9].T.flatten(),
    'equilibrium': lambda: model.equilibrium[showi % 9].T.flatten(),
    'force': lambda: model.force[showi % 2].T.flatten(),
}


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
    minvar = np.min(plotvar)
    maxvar = 1.001 * (np.max(plotvar))

    # Avoid divide by zero: maxvar == minvar <--> maxvar == minvar == 0
    maxvar = 1 if maxvar == minvar else maxvar

    # convert the plotvar array into an array of colors to plot
    # if the mesh point is solid, make it black
    frac = (plotvar[:] - minvar) / (maxvar - minvar)
    icol = frac * ncol
    plot_rgba[:] = cmap_rgba[icol.astype(np.int)]
    plot_rgba[
        model.obstacle_mask.T.flatten()] = 0xFF000000  #Color code of black

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
        drag = model.drag_coefficient
        glutSetWindowTitle("Drag {:0.3f} Time {:0.3f}".format(
            drag, model.time))
        # print()

    glutPostRedisplay()


### IO functions
def toggle_pause():
    global paused
    paused = not paused


def set_show(val):
    global show
    if val in get_var_map:
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
    # glutMouseFunc(mouse)
    # glutMotionFunc(mouse_motion)
    glutKeyboardFunc(keyboard)

    # Start main loop
    glutMainLoop()


# key to action mapping
key_action_map = {
    b'p': toggle_pause,
    b'r': lambda: model.reset(),
    b'd': lambda: set_show('density'),
    b'v': lambda: set_show('velocity'),
    b'f': lambda: set_show('population'),
    b'e': lambda: set_show('equilibrium'),
    b'g': lambda: set_show('force'),
    b'q': lambda: sys.exit(),
    b's': lambda: model.step(),
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


def run(model_):
    # Setup this module with given model.
    global model, plot_rgba, cmap_rgba, nx, ny, ncol

    model = model_
    nx, ny = model.shape
    plot_rgba = np.zeros(np.prod(model.shape), dtype=np.uint32)
    cmap_rgba, ncol = get_cmap_from_matplotlib()

    run_opengl()
