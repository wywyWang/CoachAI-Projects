# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'preprocess.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSizePolicy, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(410, 401)
        self.directory = QLineEdit(Form)
        self.directory.setObjectName(u"directory")
        self.directory.setGeometry(QRect(140, 39, 251, 21))
        self.load_directory = QPushButton(Form)
        self.load_directory.setObjectName(u"load_directory")
        self.load_directory.setGeometry(QRect(370, 41, 16, 18))
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(22, 40, 101, 20))
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 80, 111, 20))
        self.load_homography = QPushButton(Form)
        self.load_homography.setObjectName(u"load_homography")
        self.load_homography.setGeometry(QRect(370, 81, 16, 18))
        self.homography_filename = QLineEdit(Form)
        self.homography_filename.setObjectName(u"homography_filename")
        self.homography_filename.setGeometry(QRect(138, 79, 251, 21))
        self.list_filename = QLineEdit(Form)
        self.list_filename.setObjectName(u"list_filename")
        self.list_filename.setGeometry(QRect(140, 118, 251, 21))
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(22, 119, 61, 20))
        self.load_list = QPushButton(Form)
        self.load_list.setObjectName(u"load_list")
        self.load_list.setGeometry(QRect(370, 120, 16, 18))
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(22, 161, 91, 20))
        self.output_filename = QLineEdit(Form)
        self.output_filename.setObjectName(u"output_filename")
        self.output_filename.setGeometry(QRect(140, 160, 251, 21))
        self.progressBar = QProgressBar(Form)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(30, 230, 351, 23))
        self.progressBar.setValue(0)
        self.confirm = QPushButton(Form)
        self.confirm.setObjectName(u"confirm")
        self.confirm.setGeometry(QRect(320, 270, 75, 24))
        self.error_message = QLabel(Form)
        self.error_message.setObjectName(u"error_message")
        self.error_message.setGeometry(QRect(60, 270, 241, 20))
        self.error_message.setLayoutDirection(Qt.LeftToRight)
        self.error_message.setTextFormat(Qt.AutoText)
        self.error_message.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.progress_message = QLabel(Form)
        self.progress_message.setObjectName(u"progress_message")
        self.progress_message.setGeometry(QRect(30, 210, 311, 20))
        self.progress_message.setLayoutDirection(Qt.LeftToRight)
        self.progress_message.setTextFormat(Qt.AutoText)
        self.progress_message.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.homography_filename.raise_()
        self.directory.raise_()
        self.load_directory.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.load_homography.raise_()
        self.list_filename.raise_()
        self.label_3.raise_()
        self.load_list.raise_()
        self.label_4.raise_()
        self.output_filename.raise_()
        self.progressBar.raise_()
        self.confirm.raise_()
        self.error_message.raise_()
        self.progress_message.raise_()

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.load_directory.setText(QCoreApplication.translate("Form", u"...", None))
        self.label.setText(QCoreApplication.translate("Form", u"data directory", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"homography file", None))
        self.load_homography.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"index file", None))
        self.load_list.setText(QCoreApplication.translate("Form", u"...", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"output filename", None))
        self.progressBar.setFormat(QCoreApplication.translate("Form", u"%p%", None))
        self.confirm.setText(QCoreApplication.translate("Form", u"confirm", None))
        self.error_message.setText("")
        self.progress_message.setText("")
    # retranslateUi

