import matplotlib.backends.backend_agg as agg
from PySide6.QtWidgets import (QGraphicsScene,
                               QGraphicsLineItem,
                               QGraphicsEllipseItem,
                               QGraphicsTextItem,
                               QGraphicsItemGroup,
                               QGraphicsPixmapItem,
                               QGraphicsItem)
from PySide6.QtGui import QColor, QPen, QPixmap, QImage, QBrush, QFont
from PySide6.QtCore import Qt, Signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
import torch.distributions as D

import matplotlib.colors as mcolors

Cmap = mcolors.LinearSegmentedColormap

class GMM:
    def __init__(self, parameters: tuple, type):
        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        self.weights = torch.tensor(parameters[0])
        self.means = torch.tensor(parameters[1])
        self.stdevs = torch.tensor(parameters[2])
        if type == 'landing':
            negative_count = 0
            positive_count = 0
            for i in range(3):
                if self.means[:,i,1] < 0:
                    negative_count += 1
                else:
                    positive_count += 1

            for i in range(3):
                if negative_count > positive_count:
                    if self.means[:,i,1] > 0:
                        self.means[:,i,1] *= -1
                else:
                    if self.means[:,i,1] < 0:
                        self.means[:,i,1] *= -1
        self.means[:, :, 0] = self.means[:, :, 0] * std_x + mean_x
        self.means[:, :, 1] = self.means[:, :, 1] * std_y + mean_y
        self.stdevs[:, :, 0] = self.stdevs[:, :, 0] * std_x
        self.stdevs[:, :, 1] = self.stdevs[:, :, 1] * std_y

    def mean(self):
        mean_x = self.means[:, 0].detach().numpy()
        mean_y = self.means[:, 1].detach().numpy()
        return mean_x, mean_y

    def prob(self, grid_tensor: torch.Tensor):
        # Evaluate the GMM at each point on the meshgrid
        with torch.no_grad():
            mix = D.Categorical(self.weights)
            comp = D.Independent(D.Normal(self.means, self.stdevs), 1)
            gmm = D.MixtureSameFamily(mix, comp)
            probs = gmm.log_prob(grid_tensor).exp()

            return probs

class GMMDrawer:
    def __init__(self):
        # Generate a meshgrid for visualization
        self.x0 = 50
        self.x1 = 305
        self.y0 = 150
        self.y1 = 810
        plt_x = np.linspace(self.x0, self.x1, int((self.x1 - self.x0)/2))
        plt_y = np.linspace(self.y0, self.y1, int((self.y1 - self.y0)/2))
        self.X, self.Y = np.meshgrid(plt_x, plt_y)
        grid = np.stack((self.X, self.Y), axis=2)
        self.grid_tensor = torch.tensor(grid, dtype=torch.float32)

        self.landing_color = Cmap.from_list(
            'landing', ['#198964', '#ffd700'], N=256)
        self.movement_color = Cmap.from_list(
            'movement', [(25/255, 137/255, 100/255, 0), '#0000ff'], N=256) # (0,0,0,0) is transparent

    @staticmethod
    def initFig():
        fig, ax = plt.subplots(figsize=(255/60, 660/60), dpi=80)
        ax.set_facecolor((0/255, 127/255, 102/255))
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        ax.margins(x=0, y=0)
        # turn off the border line
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim((50, 305))
        # note: invert y coord since the (0, 0) is on left top for screen coord system
        ax.set_ylim((810, 150))

        return fig, ax

    @staticmethod
    def fig2img(fig: plt.Figure):
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            size[::-1] + (3,))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        return img

    def contour_continuous(self, landing_params: tuple, movement_params: tuple) -> np.ndarray:
        if len(landing_params) == 0 or len(movement_params) == 0:
            return None
        landing_gmm = GMM(landing_params, 'landing')
        movement_gmm = GMM(movement_params, 'moving')
        fig, ax = self.initFig()
        landing_probs = landing_gmm.prob(self.grid_tensor)
        movement_probs = movement_gmm.prob(self.grid_tensor)
        plt.imshow(landing_probs, extent=[50, 305, 150, 810],
                   origin='lower', cmap=self.landing_color)
        plt.imshow(movement_probs, extent=[50, 305, 150, 810],
                   origin='lower', cmap=self.movement_color)
        # plt.scatter(mean_x, mean_y, c='red',
        #            marker='x', label='Component Means')
        # ax.set_aspect('equal', adjustable='box')
        img = self.fig2img(fig)

        return img

class Field(QGraphicsScene):
    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__()

        # distribution image
        self.pixmap = QGraphicsPixmapItem(QPixmap())
        self.pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.addItem(self.pixmap)

        # init field border
        scale = 1/1.33
        self.scale = scale
        # real size of badminton field in cm
        outer_height = 660*scale
        outer_width = 295*scale
        inner_height = 590*scale
        inner_width = 255*scale
        outer_h_start = 150*scale
        outer_h_end = outer_h_start+outer_height
        outer_w_start = 22.5*scale
        outer_w_end = outer_w_start + outer_width
        inner_height_gap = (outer_height - inner_height)/2
        inner_w_gap = (outer_width - inner_width)/2
        inner_h_start = outer_h_start + inner_height_gap
        inner_h_end = outer_h_end - inner_height_gap
        inner_w_start = outer_w_start+inner_w_gap
        inner_w_end = outer_w_end-inner_w_gap
        height_mid = (outer_h_start+outer_h_end)/2
        width_mid = (outer_w_start+outer_w_end)/2

        serve_line1 = height_mid - 114*scale
        serve_line2 = height_mid + 114*scale

        # line(start_x, start_y, end_x, end_y) coord pair for drawing field border
        line_coords = [(outer_w_start, outer_h_start, outer_w_start, outer_h_end),
                       (outer_w_start, outer_h_start, outer_w_end, outer_h_start),
                       (outer_w_start, outer_h_end, outer_w_end, outer_h_end),
                       (outer_w_end, outer_h_start, outer_w_end, outer_h_end),
                       # inner line
                       (inner_w_start, outer_h_start, inner_w_start, outer_h_end),
                       (outer_w_start, inner_h_start, outer_w_end, inner_h_start),
                       (outer_w_start, inner_h_end, outer_w_end, inner_h_end),
                       (inner_w_end, outer_h_start, inner_w_end, outer_h_end),
                       # net
                       (outer_w_start, height_mid, outer_w_end, height_mid),
                       # serve line
                       (outer_w_start, serve_line1, outer_w_end, serve_line1),
                       (outer_w_start, serve_line2, outer_w_end, serve_line2),
                       # mid line
                       (width_mid, outer_h_start, width_mid, serve_line1),
                       (width_mid, serve_line2, width_mid, outer_h_end),
                       ]

        pen = QPen(QColor(255, 255, 255), 2*self.scale)  # width: 4cm
        for coord in line_coords:
            line = QGraphicsLineItem()
            line.setPen(pen)
            line.setLine(*coord)
            self.addItem(line)
        #self.update()
        discrete_boundary_coords = [
            # horizontal line
            (inner_w_start, outer_h_start + outer_height/6,   inner_w_end, outer_h_start + outer_height/6),
            (inner_w_start, outer_h_start + outer_height/6*2, inner_w_end, outer_h_start + outer_height/6*2),
            # (inner_width_start, outer_height/6*3, inner_width_end, outer_height/6*3), # net
            (inner_w_start, outer_h_start + outer_height/6*4, inner_w_end, outer_h_start + outer_height/6*4),
            (inner_w_start, outer_h_start + outer_height/6*5, inner_w_end, outer_h_start + outer_height/6*5),
            # vertical line
            (inner_w_start + inner_width/3,   outer_h_start, inner_w_start + inner_width/3, outer_h_start+outer_height),
            (inner_w_start + inner_width/3*2, outer_h_start, inner_w_start + inner_width/3*2, outer_h_start+outer_height)
        ]
        pen = QPen(QColor(37, 207, 152), 1*self.scale)  # width: 4cm
        for coord in discrete_boundary_coords:
            line = QGraphicsLineItem()
            line.setPen(pen)
            line.setLine(*coord)
            self.addItem(line)

        self.ball = QGraphicsEllipseItem(0, 0, 10, 10)
        self.ball.setPen(QColor())
        self.ball.setBrush(QColor(0, 0, 255, 0))
        self.setBallPos(170, 480)
        self.addItem(self.ball)

        self.playerA_color = QColor(255, 127, 39)
        self.playerB_color = QColor(255, 0, 255)
        self.playerA = self.objectInit(self.playerA_color, 'A')
        self.playerB = self.objectInit(self.playerB_color, 'B')
        self.setPlayerBPos(170, 360)
        self.setPlayerAPos(170, 600)

        self.text_fieldA_probability = []
        self.text_fieldB_probability = []

        self.show_probability_text = True
        self.initProbabilityText()

        self.currentLauncher = 'A'

        self.gmm_drawer = GMMDrawer() 

    def initProbabilityText(self):
        #9 8 7
        #6 5 4
        #3 2 1
        self.text_fieldA_coord = [(262.5 *self.scale, 425*self.scale),
                                  (177.5 *self.scale, 425*self.scale),
                                  (92.5  *self.scale, 425*self.scale),
                                  (262.5 *self.scale, 315*self.scale),
                                  (177.5 *self.scale, 315*self.scale),
                                  (92.5  *self.scale, 315*self.scale),
                                  (262.5 *self.scale, 205*self.scale),
                                  (177.5 *self.scale, 205*self.scale),
                                  (92.5  *self.scale, 205*self.scale)]
        for i in range(1,10):
            self.text_fieldA_probability.append(self.addTextObject(i,self.text_fieldA_coord))
        self.showFieldAProbability()

        #1 2 3
        #4 5 6
        #7 8 9
        self.text_fieldB_coord= [(92.5  *self.scale, 535*self.scale),
                                 (177.5 *self.scale, 535*self.scale),
                                 (262.5 *self.scale, 535*self.scale),
                                 (92.5  *self.scale, 645*self.scale),
                                 (177.5 *self.scale, 645*self.scale),
                                 (262.5 *self.scale, 645*self.scale),
                                 (92.5  *self.scale, 755*self.scale),
                                 (177.5 *self.scale, 755*self.scale),
                                 (262.5 *self.scale, 755*self.scale)]
        for i in range(1, 10):
            self.text_fieldB_probability.append(self.addTextObject(i,self.text_fieldB_coord))
        self.showFieldBProbability()

    def centerize(self, item: QGraphicsItem, x: float, y: float):
        width = item.boundingRect().width()  # get text width
        height = item.boundingRect().height()  # get text height
        item.setPos(x - width/2, y - height/2)

    def addTextObject(self, i: int, coord: list):
        text = QGraphicsTextItem(f'{i}')
        text.setDefaultTextColor(QColor(37, 207, 152))
        text.setFont(QFont('Arial', 12))
        self.centerize(text, *coord[i-1])
        self.addItem(text)
        return text

    def setDistributionImg(self, isDisplay, landing_prob, movement_prob):
        if isDisplay:
            img = self.gmm_drawer.contour_continuous(landing_prob, movement_prob)
        else:
            img = None
        self.setBackgroundImage(img)

    def setBackgroundImage(self, image: np.ndarray):
        if image is None:
            self.pixmap.setPixmap(QPixmap())
            return
        # print(image.shape, image)
        qimage = QImage(image.data, image.shape[1], image.shape[0],
                        image.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap.setPixmap(pixmap)
        height = self.pixmap.boundingRect().height()
        self.pixmap.setScale(660/height*self.scale)
        self.pixmap.setPos(50*self.scale, 150*self.scale)
        # print(self.pixmap.boundingRect().height())
        # self.pixmap.update()

    # setup color and text for player
    def objectInit(self, color: QColor, text: str):
        group = QGraphicsItemGroup()
        obj = QGraphicsEllipseItem(0, 0, 12, 12)
        obj.setBrush(color)
        obj.setPen(color)
        group.addToGroup(obj)

        text_object = QGraphicsTextItem(text)
        text_object.setPos(-2, -6)
        group.addToGroup(text_object)

        self.addItem(group)
        return group

    def showFieldAProbability(self):
        if self.show_probability_text:
            for text in self.text_fieldA_probability:
                text.show()
        #for text in self.text_fieldB_probability:
        #    text.hide()

    def showFieldBProbability(self):
        #for text in self.text_fieldA_probability:
        #    text.hide()
        if self.show_probability_text:
            for text in self.text_fieldB_probability:
                text.show()


    # color for ball should same as launcher
    def changeBallLauncher(self, player: str):
        self.currentLauncher = player
        if player == 'A':
            self.ball.setPen(self.playerA_color)
            #self.showFieldAProbability()
        elif player == 'B':
            self.ball.setPen(self.playerB_color)
            #self.showFieldBProbability()

    def coordScale(self, coord):
        return (coord[0]*self.scale, coord[1]*self.scale)

    def setObjectPos(self, obj: QGraphicsItem, x: float, y: float):
        x, y = self.coordScale((x, y))
        x -= 12/2
        y -= 12/2
        obj.setPos(x, y)

    def setBallPos(self, x: float, y: float):
        self.setObjectPos(self.ball, x, y)

    def setPlayerAPos(self, x: float, y: float):
        self.setObjectPos(self.playerA, x, y)

    def setPlayerBPos(self, x: float, y: float):
        self.setObjectPos(self.playerB, x, y)

    def setBallScale(self, scale: float):
        self.ball.setScale(scale)

    # add click event
    def mouseReleaseEvent(self, event):
        x = event.scenePos().x()/self.scale
        y = event.scenePos().y()/self.scale
        # print(x, y)
        self.clicked.emit(x, y)
        super().mouseReleaseEvent(event)