# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'generate.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QLabel,
    QLineEdit, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QToolButton, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(470, 466)
        self.model1_choose = QGroupBox(Form)
        self.model1_choose.setObjectName(u"model1_choose")
        self.model1_choose.setGeometry(QRect(30, 10, 371, 80))
        self.model1_ShuttleNet = QRadioButton(self.model1_choose)
        self.model1_ShuttleNet.setObjectName(u"model1_ShuttleNet")
        self.model1_ShuttleNet.setGeometry(QRect(20, 20, 161, 20))
        self.model1_ShuttleNet.setChecked(True)
        self.model1_custom = QRadioButton(self.model1_choose)
        self.model1_custom.setObjectName(u"model1_custom")
        self.model1_custom.setGeometry(QRect(20, 50, 91, 20))
        self.model1_custom_path = QLineEdit(self.model1_choose)
        self.model1_custom_path.setObjectName(u"model1_custom_path")
        self.model1_custom_path.setEnabled(False)
        self.model1_custom_path.setGeometry(QRect(110, 50, 251, 20))
        self.model1_load_custom = QToolButton(self.model1_choose)
        self.model1_load_custom.setObjectName(u"model1_load_custom")
        self.model1_load_custom.setEnabled(False)
        self.model1_load_custom.setGeometry(QRect(340, 52, 16, 16))
        self.model1_ShuttleNet_player = QComboBox(self.model1_choose)
        self.model1_ShuttleNet_player.addItem("")
        self.model1_ShuttleNet_player.setObjectName(u"model1_ShuttleNet_player")
        self.model1_ShuttleNet_player.setGeometry(QRect(200, 20, 161, 22))
        self.model2_choose = QGroupBox(Form)
        self.model2_choose.setObjectName(u"model2_choose")
        self.model2_choose.setEnabled(True)
        self.model2_choose.setGeometry(QRect(30, 100, 371, 80))
        self.model2_ShuttleNet = QRadioButton(self.model2_choose)
        self.model2_ShuttleNet.setObjectName(u"model2_ShuttleNet")
        self.model2_ShuttleNet.setGeometry(QRect(20, 20, 161, 20))
        self.model2_ShuttleNet.setChecked(True)
        self.model2_custom = QRadioButton(self.model2_choose)
        self.model2_custom.setObjectName(u"model2_custom")
        self.model2_custom.setGeometry(QRect(20, 50, 91, 20))
        self.model2_custom_path = QLineEdit(self.model2_choose)
        self.model2_custom_path.setObjectName(u"model2_custom_path")
        self.model2_custom_path.setEnabled(False)
        self.model2_custom_path.setGeometry(QRect(110, 50, 251, 20))
        self.model2_load_custom = QToolButton(self.model2_choose)
        self.model2_load_custom.setObjectName(u"model2_load_custom")
        self.model2_load_custom.setEnabled(False)
        self.model2_load_custom.setGeometry(QRect(340, 52, 16, 16))
        self.model2_ShuttleNet_player = QComboBox(self.model2_choose)
        self.model2_ShuttleNet_player.addItem("")
        self.model2_ShuttleNet_player.setObjectName(u"model2_ShuttleNet_player")
        self.model2_ShuttleNet_player.setGeometry(QRect(200, 20, 161, 22))
        self.confirm = QPushButton(Form)
        self.confirm.setObjectName(u"confirm")
        self.confirm.setGeometry(QRect(320, 280, 75, 24))
        self.error_message = QLabel(Form)
        self.error_message.setObjectName(u"error_message")
        self.error_message.setGeometry(QRect(190, 240, 201, 20))
        self.error_message.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.output_filename = QLineEdit(Form)
        self.output_filename.setObjectName(u"output_filename")
        self.output_filename.setGeometry(QRect(110, 200, 281, 20))
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(50, 190, 53, 41))
        self.label.setAlignment(Qt.AlignCenter)
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(50, 240, 71, 21))
        self.rally_count = QLineEdit(Form)
        self.rally_count.setObjectName(u"rally_count")
        self.rally_count.setGeometry(QRect(120, 240, 61, 20))
        self.progressBar = QProgressBar(Form)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(50, 280, 261, 23))
        self.progressBar.setValue(0)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.model1_choose.setTitle(QCoreApplication.translate("Form", u"Agent 1", None))
        self.model1_ShuttleNet.setText(QCoreApplication.translate("Form", u"Default agent ", None))
        self.model1_custom.setText(QCoreApplication.translate("Form", u"Other agent", None))
        self.model1_load_custom.setText(QCoreApplication.translate("Form", u"...", None))
        self.model1_ShuttleNet_player.setItemText(0, QCoreApplication.translate("Form", u"(choose the opponent)", None))

        self.model2_choose.setTitle(QCoreApplication.translate("Form", u"Agent 2", None))
        self.model2_ShuttleNet.setText(QCoreApplication.translate("Form", u"Default agent ", None))
        self.model2_custom.setText(QCoreApplication.translate("Form", u"Other agent", None))
        self.model2_load_custom.setText(QCoreApplication.translate("Form", u"...", None))
        self.model2_ShuttleNet_player.setItemText(0, QCoreApplication.translate("Form", u"(choose the opponent)", None))

        self.confirm.setText(QCoreApplication.translate("Form", u"generate", None))
        self.error_message.setText("")
        self.label.setText(QCoreApplication.translate("Form", u"Ouput\n"
"Filename", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Rally Count", None))
        self.rally_count.setText(QCoreApplication.translate("Form", u"1000", None))
    # retranslateUi

