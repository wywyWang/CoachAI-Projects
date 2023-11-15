from PySide6.QtWidgets import QWidget
from GUI.stimulate_ui import Ui_Form as StimulateWidgetUi

class StimulateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = StimulateWidgetUi()
        self.ui.setupUi(self)