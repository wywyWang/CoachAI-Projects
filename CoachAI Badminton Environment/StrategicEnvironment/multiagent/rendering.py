"""
2D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

#from gym.utils import reraise
from gym import error
import pyglet
from pyglet.gl import *
'''
try:
    
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
   
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")
'''
import math
import numpy as np

RAD2DEG = 57.29577951308232

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))
        

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=display, resizable=True)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.window.distribution = None 

        self.window.wholabel = pyglet.text.Label("A     B",
                                font_name='Times New Roman',
                                font_size=10,
                                color=(0,0,0,255),
                                x=100, y=870,
                                anchor_x='center', anchor_y='center')
        self.window.score = None
        self.window.ball_info = None
        self.window.playerA_label = pyglet.text.Label("A",
                                    font_name='Times New Roman',
                                    font_size=20,
                                    color=(0,0,0,255),
                                    x=20, y=700,
                                    anchor_x='center', anchor_y='center')
        self.window.playerB_label = pyglet.text.Label("B",
                                    font_name='Times New Roman',
                                    font_size=20,
                                    color=(0,0,0,255),
                                    x=20, y=300,
                                    anchor_x='center', anchor_y='center')

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, score=None, return_rgb_array=False, ball_info = None, end_reason = None, ball_prob = None):
        if end_reason is None or end_reason == 0:
            end_reason = None
        glClearColor(1,1,1,1)
       
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        
        #pyglet.app.run()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()

        #if ball_prob is not None and len(ball_prob) == 10:
        #    distribution = f'發短球  {ball_prob[0]*100:.2f}%\n'\
        #                   f'長球　  {ball_prob[1]*100:.2f}%\n'\
        #                   f'推球　  {ball_prob[2]*100:.2f}%\n'\
        #                   f'殺球　  {ball_prob[3]*100:.2f}%\n'\
        #                   f'擋小球  {ball_prob[4]*100:.2f}%\n'\
        #                   f'平球　  {ball_prob[5]*100:.2f}%\n'\
        #                   f'放小球  {ball_prob[6]*100:.2f}%\n'\
        #                   f'挑球　  {ball_prob[7]*100:.2f}%\n'\
        #                   f'切球　  {ball_prob[8]*100:.2f}%\n'\
        #                   f'發長球  {ball_prob[9]*100:.2f}%'
        #else:
        #    distribution = f'發短球  ----%\n'\
        #                   f'長球　  ----%\n'\
        #                   f'推球　  ----%\n'\
        #                   f'殺球　  ----%\n'\
        #                   f'擋小球  ----%\n'\
        #                   f'平球　  ----%\n'\
        #                   f'放小球  ----%\n'\
        #                   f'挑球　  ----%\n'\
        #                   f'切球　  ----%\n'\
        #                   f'發長球  ----%'
#
        #if distribution != self.window.distribution:
        #    self.window.distribution_label = pyglet.text.Label(distribution,
        #                        font_name='Times New Roman',
        #                        font_size=15,
        #                        color=(0,0,0,255),
        #                        x=350, y=500,
        #                        anchor_x='left', anchor_y='center', multiline = True, width = 200)
        #    self.window.distribution = distribution
        #self.window.distribution_label.draw()
        if score is not None:
            self.window.wholabel.draw()

            if score != self.window.score:
                score = str(score[0])+":"+str(score[1])
                self.window.score = score
                self.window.score_label = pyglet.text.Label(score,
                                font_name='Times New Roman',
                                font_size=30,
                                color=(0,0,0,255),
                                x=100, y=850,
                                anchor_x='center', anchor_y='center')
            self.window.score_label.draw()

        
        if ball_info is not None and ball_info[1] is not None:
            #ball_type should be a string 
            if ball_info[1] == 0:
                ball_info = (ball_info[0], '未知球種')

            if ball_info != self.window.ball_info:
                player, ball_type = ball_info
                print(f'player{player}')
                self.window.ball_info = ball_info
                self.window.ball_label = pyglet.text.Label(ball_type,
                                    font_name='Arial',
                                    font_size=20,
                                    color=(0,0,0,255),
                                    x=250, y=850 if player == 1 else 100,
                                    anchor_x='center', anchor_y='center')
            self.window.ball_label.draw()

        self.window.playerA_label.draw()
        self.window.playerB_label.draw()

        if end_reason is not None:
            self.window.label = pyglet.text.Label(end_reason,
                                font_name='Arial',
                                font_size=20,
                                color=(255,0,0,255),
                                x=175, y=130,
                                anchor_x='center', anchor_y='center')
            self.window.label.draw()
        
        #print(score)
        '''y = 900
        for key, value in self.info.items():
            self.window.label = pyglet.text.Label(str(key)+':'+str(value),
                                font_name='Times New Roman',
                                font_size=10,
                                color=(0,0,0,255),
                                x=300, y=y,
                                anchor_x='center', anchor_y='center')
            self.window.label.draw()
            y-=50'''
            
        
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

        color = (self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom



class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
        
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
        
    def set_linewidth(self, x):
        print('set')
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        #self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)
        self.img.blit(30, 150, width=295, height=660)

'''
load image from array
'''
class ImageArr(Geom):
    def __init__(self, arr, x, y, width, height, scale_width = None, scale_height = None):
        Geom.__init__(self)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale_width = scale_width
        self.scale_height = scale_height
        if self.scale_width is None:
            self.scale_width = width
        if self.scale_height is None:
            self.scale_height = height

        img = pyglet.image.ImageData(self.width, self.height, 'BGR', arr.tobytes(), pitch=self.width * -3)
        self.img = img
        self.flip = False
    def render1(self):
        #self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)
        self.img.blit(self.x, self.y, width=self.scale_width, height=self.scale_height)

    def changeImage(self, arr, width, height):
        self.width = width
        self.height = height
        self.img = pyglet.image.ImageData(self.width, self.height, 'BGR', arr.tobytes(), pitch=self.width * -3)
        self.flip = False

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()