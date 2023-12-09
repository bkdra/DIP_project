from window import Ui_MainWindow
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
from fittingRoom import FittingRoom

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.ft = FittingRoom()
        self.pictureModeBtn.clicked.connect(self.pictureModeBtnClicked)
        self.cameraModeBtn.clicked.connect(self.cameraModeBtnClicked)
        self.selectClothbtn.clicked.connect(self.selectClothBtnClicked)
        self.selectPantsbtn.clicked.connect(self.selectPantsBtnClicked)
        self.selectSunglassesbtn.clicked.connect(self.selectSunglassesBtnClicked)
        self.pantSizeSlider.valueChanged.connect(self.pantSizeSliderChanged)
        self.clothSizeSlider.valueChanged.connect(self.clothSizeSliderChanged)
        self.set_style()
        
    
    def pictureModeBtnClicked(self):
        self.pictureModeBtn.setChecked(True)
        self.cameraModeBtn.setChecked(False)
        self.ft.picPath = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.update_picture()
        except:
            pass

    def cameraModeBtnClicked(self):
        self.pictureModeBtn.setChecked(False)
        self.cameraModeBtn.setChecked(True)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.label.setText("攝影機未開啟")
            return
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # 每秒30幀
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            processed_frame = self.ft.main_capture(frame)
            # 將 OpenCV 影像轉換為 QImage
            cvRGBImg = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.imageLabel.setPixmap(pixmap)

    def selectClothBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_cloth(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass

    def selectPantsBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_pants(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass
    
    def selectSunglassesBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            self.ft.set_sunglass(path)
            if self.pictureModeBtn.isChecked():
                self.update_picture()
        except:
            pass

    def pantSizeSliderChanged(self):
        self.ft.set_pantsSize(self.pantSizeSlider.value()/100.0)
        if self.pictureModeBtn.isChecked():
            self.update_picture()

    def clothSizeSliderChanged(self):
        self.ft.set_clothSize(self.clothSizeSlider.value()/100.0)
        print(self.clothSizeSlider.value())
        if self.pictureModeBtn.isChecked():
            self.update_picture()
    
    def update_picture(self):
        img = self.ft.main_picture()
        cvRGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.imageLabel.setPixmap(pixmap)
    
    def set_style(self):
        SBstylesheet = '''
            QScrollBarArea{
                border: 0px;
           }
            QScrollBar:vertical
            {
                background : solid rgb(31,41,55);
            }

            QScrollBar::handle:vertical
            {
                background-color: rgb(50, 130, 184)
            }

            QScrollBar::groove:vertical
            {
                background-color: rgb(31,41,55);
            }
            QScrollBar:horizontal
            {
                background : solid rgb(31,41,55);
            }

            QScrollBar::handle:horizontal
            {
                background-color: rgb(50, 130, 184)
            }

            QScrollBar::groove:horizontal
            {
                background-color: rgb(31,41,55);
            }
            ''' 
        self.imageArea.setStyleSheet(SBstylesheet)
        modeBtnstylesheet = '''
            QPushButton{
                color: rgb(0, 0, 0);
                font: 75 10pt "Consolas";
                border-color: rgb(187, 225, 250);
                border: 2px solid rgb(187, 225, 250);
                border-top-left-radius :7px;
                background-color: rgb(187, 225, 250);
                border-top-right-radius : 7px;                  
                border-bottom-left-radius : 7px;
                border-bottom-right-radius : 7px
            }
            QPushButton:hover{
                background-color: rgb(50, 130, 184);
                border: 2px solid rgb(50, 130, 184);
            }
            QPushButton:checked{
                background-color: rgb(15, 76, 117);
                border: 2px solid rgb(15, 76, 117);
            }
        '''
        self.pictureModeBtn.setStyleSheet(modeBtnstylesheet)
        self.cameraModeBtn.setStyleSheet(modeBtnstylesheet)

        selectBtnstylesheet = '''
            QPushButton{
                color: rgb(0, 0, 0);
                font: 75 10pt "Consolas";
                border-color: rgb(187, 225, 250);
                border: 2px solid rgb(187, 225, 250);
                border-top-left-radius :7px;
                background-color: rgb(187, 225, 250);
                border-top-right-radius : 7px;                  
                border-bottom-left-radius : 7px;
                border-bottom-right-radius : 7px
            }
            QPushButton:hover{
                background-color: rgb(50, 130, 184);
                border: 2px solid rgb(50, 130, 184);
            }
        '''
        self.selectClothbtn.setStyleSheet(selectBtnstylesheet)
        self.selectPantsbtn.setStyleSheet(selectBtnstylesheet)
        self.selectSunglassesbtn.setStyleSheet(selectBtnstylesheet)