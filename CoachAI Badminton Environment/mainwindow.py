import sys
import seaborn as sns
from PySide6.QtWidgets import QApplication,QMainWindow
from PySide6.QtCore import Qt, Signal
from GUI.mainwindow_ui import Ui_MainWindow
from GUI.PreprocessWidget import PreprocessWidget
from GUI.StimulateWidget import StimulateWidget
from GUI.TrainingWidget import TrainingWidget
from GUI.GenerateWidget import GenerateWidget
from GUI.field import Field
from preprocessTabCtrl import PreprocessWidgetCtrl
from stimulateTabCtrl import StimulateWidgetCtrl
from trainingTabCtrl import TrainingWidgetCtrl
from generateWidgetCtrl import GenerateWidgetCtrl

"""
The frame of this application
Contained a field that can show position animationally,
and tabs with different function
Each tabs is composed of two parts, widget and controller,
we'll initial the widget first and pass it to a controller to control it
"""
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.field = Field()
        self.ui.field.setScene(self.field)
        self.ui.field.setSceneRect(self.field.sceneRect())

        # init each tabs widgets
        #self.preprocess_widget = PreprocessWidget(self)
        self.stimulate_widget = StimulateWidget(self)
        #self.training_widget = TrainingWidget(self)
        self.generate_widget = GenerateWidget(self)
        #self.ui.tabWidget.addTab(self.preprocess_widget, 'Preprocess')
        #self.ui.tabWidget.addTab(self.training_widget, 'Training')
        self.ui.tabWidget.addTab(self.generate_widget, 'Generate')
        self.ui.tabWidget.addTab(self.stimulate_widget, 'Visualize')
        
        # move widgets in current window to tabs that need them
        self.stimulate_widget.ui.field = self.ui.field
        self.stimulate_widget.ui.score = self.ui.score
        self.stimulate_widget.ui.playerA_type = self.ui.playerA_type
        self.stimulate_widget.ui.playerB_type = self.ui.playerB_type
        self.stimulate_widget.ui.ball_round = self.ui.ball_round

        # pass widgets to their controller
        #self.preprocess_widget_ctrl = PreprocessWidgetCtrl(self.preprocess_widget)
        self.stimulate_widget_ctrl = StimulateWidgetCtrl(self.stimulate_widget)
        #self.training_widget_ctrl = TrainingWidgetCtrl(self.training_widget)
        self.generate_widget_ctrl = GenerateWidgetCtrl(self.generate_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())