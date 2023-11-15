from PySide6.QtWidgets import QWidget
from GUI.preprocess_ui import Ui_Form as PreprocessWidgetUi

class PreprocessWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = PreprocessWidgetUi()
        self.ui.setupUi(self)