from PySide6.QtWidgets import QWidget
from GUI.training_ui import Ui_Form as TrainingWidgetUi

class TrainingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = TrainingWidgetUi()
        self.ui.setupUi(self)