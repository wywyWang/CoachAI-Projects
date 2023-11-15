from PySide6.QtWidgets import QWidget
from GUI.generate_ui import Ui_Form as GenerateWidgetUi

class GenerateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = GenerateWidgetUi()
        self.ui.setupUi(self)