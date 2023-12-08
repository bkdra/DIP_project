from window import Ui_fittingRoom
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtGui import QImage, QPixmap
import cv2
import fittingRoom

class MainWindow(QMainWindow,Ui_fittingRoom):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.pictureModeBtn.clicked.connect(self.pictureModeBtnClicked)
        self.cameraModeBtn.clicked.connect(self.cameraModeBtnClicked)
    
    def pictureModeBtnClicked(self):
        path = QFileDialog.getOpenFileName(self, "開啟圖片", "./")[0]
        try:
            img = fittingRoom.main_picture(path)       
            cvRGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)
        except:
            pass

    def cameraModeBtnClicked(self):
        fittingRoom.main_capture()
            
