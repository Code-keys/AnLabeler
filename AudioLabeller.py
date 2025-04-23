import os
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QSlider, QMessageBox,
                            QInputDialog, QListWidget, QListWidgetItem, QSplitter, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
import pyqtgraph as pg
import soundfile as sf
import librosa
import librosa.display
from scipy.io import wavfile

import glob

def find_wav_files(folder_path):
    """使用glob查找所有WAV文件（包括子文件夹）"""
    return glob.glob(os.path.join(folder_path, '**/*.wav'), recursive=True)

class SyncViewBox(pg.ViewBox):
    """同步缩放视图"""
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.linked_views = []
    
    def linkView(self, view):
        if view not in self.linked_views:
            self.linked_views.append(view)

    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        return
        if ev.isAccepted():
            for view in self.linked_views:
                view.setXRange(*self.viewRange()[0], padding=0)

    def mouseScrollEvent(self, ev, axis=None):
        """鼠标滚轮缩放时同步所有链接的ViewBox"""
        super().mouseScrollEvent(ev, axis)
        if ev.isAccepted():
            self._sync_linked_views()

    def _sync_linked_views(self):
        """同步所有链接的ViewBox的显示范围"""
        x_range, y_range = self.viewRange()
        for view in self.linked_views:
            # 避免递归调用（防止死循环）
            if view is not self:
                view.blockSignals(True)  # 临时阻止信号
                view.setRange(xRange=x_range, yRange=y_range, padding=0)
                view.blockSignals(False)

class AudioViewer(pg.PlotWidget):
    """音频视图基类"""
    selection_changed = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        view_box = SyncViewBox()
        super().__init__(parent, viewBox=view_box)
        self.setMouseEnabled(x=True, y=False)
        self.setLabel('bottom', 'Time (s)')

        self.hideAxis('left')
        self.plotItem.showGrid(x=True, y=True)

        self.setBackground(QColor(50, 50, 50))  # 暗灰色/ # 设置坐标轴和文本颜色为浅色（适合暗背景）
        self.getAxis('left').setPen('w')  # 白色 Y 轴
        self.getAxis('bottom').setPen('w')  # 白色 X 轴
        # 设置网格线（灰色半透明）
        self.showGrid(x=True, y=True, alpha=0.3)  # 显示网格，透明度 30%

        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        self.selection_active = False
        self.playback_pos = None

    def linkView(self, view):
        self.getViewBox().linkView(view.getViewBox())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            self.selection_start = pos.x()
            self.selection_end = pos.x()
            self.selection_active = True
            self.update_selection_rect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selection_active:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            self.selection_end = pos.x()
            self.update_selection_rect()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.selection_active:
            self.selection_active = False
            if abs(self.selection_end - self.selection_start) < 0.0000001:  # 点击而非拖动
                self.clear_selection()
            else:
                self.selection_changed.emit(*self.get_selection())
        super().mouseReleaseEvent(event)
    
    def update_selection_rect(self):
        if self.selection_start is not None and self.selection_end is not None:
            x0 = min(self.selection_start, self.selection_end)
            x1 = max(self.selection_start, self.selection_end)
            y_range = self.viewRange()[1]
            
            if self.selection_rect is None:
                self.selection_rect = pg.LinearRegionItem(values=[x0, x1])
                self.selection_rect.setZValue(10)
                self.selection_rect.setBrush(QBrush(QColor(0, 0, 255, 50)))
                self.addItem(self.selection_rect)
            else:
                self.selection_rect.setRegion([x0, x1])
    
    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        if self.selection_rect:
            self.removeItem(self.selection_rect)
            self.selection_rect = None
    
    def get_selection(self):
        if self.selection_start is not None and self.selection_end is not None:
            return sorted([self.selection_start, self.selection_end])
        return None

    def set_playback_pos(self, pos):
        """设置播放位置指示器"""
        if self.playback_pos is None:
            self.playback_pos = pg.InfiniteLine(pos, angle=90, pen=pg.mkPen('r', width=2))
            self.addItem(self.playback_pos)
        else:
            self.playback_pos.setValue(pos)
    
    def clear_playback_pos(self):
        """清除播放位置指示器"""
        if self.playback_pos:
            self.removeItem(self.playback_pos)
            self.playback_pos = None

class WaveformViewer(AudioViewer):
    """波形视图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # 定义 Audition 风格的色
        # self.setLabel('top', 'Time (s)')
        audition_green = QColor(67, 217, 150)  # 纯青绿色
        self.waveform_plot = self.plot(
            pen=audition_green, width=1,
            symbol='o',  # 圆点符号
            symbolSize=4,  # 点的大小
            symbolBrush=audition_green,  # 填充颜色
            symbolPen=None  # 边框颜色（None 表示无边框）  
        )
        self.label_regions = []

        # 设置波形图和频谱图背景为黑色
        y_axis = self.getAxis('right')  # 'left' 表示左侧 Y 轴

        self.setYRange(-1, 1)  # 设置 Y 轴范围为 -10 到 10

        # self.setLogMode(y=True)  # Y 轴对数刻度

        self.setBackground(QColor(50, 50, 50))  # 暗灰色/ # 设置坐标轴和文本颜色为浅色（适合暗背景

        # 设置坐标轴和网格颜色为白色
        self.getAxis('bottom').setPen( QColor(67, 217, 150) )   # 纯青绿色
        self.getAxis('bottom').setTextPen('w')
        self.showGrid(x=True, y=True, alpha=0.3)


    def set_waveform(self, time_axis, data):
        self.waveform_plot.setData(time_axis, data)
    
    def add_label_region(self, start, end, label_text, color='g'):
        """添加标签区域"""
        region = pg.LinearRegionItem(values=[start, end])
        region.setBrush(QBrush(QColor(color)))
        region.setZValue(5)
        
        # 添加标签文本
        label = pg.TextItem(label_text, color='k', anchor=(0, 1))
        label.setPos(start, self.viewRange()[1][1] * 0.9)
        
        self.addItem(region)
        self.addItem(label)
        
        self.label_regions.append({
            'region': region,
            'label': label,
            'start': start,
            'end': end,
            'text': label_text
        })
        return len(self.label_regions) - 1
    
    def update_label_region(self, index, start=None, end=None, text=None):
        """更新标签区域"""
        if index >= len(self.label_regions):
            return
            
        item = self.label_regions[index]
        if start is not None and end is not None:
            item['region'].setRegion([start, end])
            item['start'] = start
            item['end'] = end
            item['label'].setPos(start, self.viewRange()[1][1] * 0.9)
        
        if text is not None:
            item['label'].setText(text)
            item['text'] = text
    
    def remove_label_region(self, index):
        """移除标签区域"""
        if index >= len(self.label_regions):
            return

        item = self.label_regions.pop(index)
        self.removeItem(item['region'])
        self.removeItem(item['label'])
    
    def clear_label_regions(self):
        """清除所有标签区域"""
        for item in self.label_regions:
            self.removeItem(item['region'])
            self.removeItem(item['label'])
        self.label_regions = []

class SpectrogramViewer(pg.PlotWidget):
    """频谱图视图"""
    def __init__(self, parent=None):
        view_box = SyncViewBox()
        super().__init__(parent, viewBox=view_box)
        
        # 设置背景和坐标轴
        self.setBackground(QColor(30, 30, 30))  # 更暗的背景
        
        self.hideAxis('left')
        # 创建图像项
        self.img = pg.ImageItem()
        self.addItem(self.img)
        
        # 使用更适合音频的色图 magma plasma
        self.set_colormap('magma')
        
        # 初始化显示范围
        self.getViewBox().setAspectLocked(False)
        self.getViewBox().setMouseEnabled(x=True, y=True)

    def set_colormap(self, name='plasma'):
        """设置颜色映射"""
        cmap = pg.colormap.get(name)
        self.img.setLookupTable(cmap.getLookupTable(alpha=False))

    def linkView(self, view):
        """链接其他视图"""
        self.getViewBox().linkView(view.getViewBox())

    def set_spectrogram(self, data, extent=None):
        """
        设置频谱图数据
        :param data: 2D numpy数组 (freq_bins, time_frames)
        :param extent: [xmin, xmax, ymin, ymax] 坐标范围
        """
        # 数据预处理
        if data.ndim != 2:
            raise ValueError("频谱数据必须是2D数组")

        # 设置图像数据
        self.img.setImage(data.T, autoLevels=True)  # 转置使时间轴在x方向

        # 设置坐标范围
        # if extent is not None:
        #     self.img.setRect(QRectF(*extent))
        # else:
        #     # 自动计算范围
        h, w = data.shape
        self.img.setRect(QRectF(0, 0, w, h))

        # 自动调整色阶
        self.img.setLevels([data.min(), data.max()])
        
        # 刷新显示
        self.getViewBox().autoRange()

class AudioLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyAudioLabeler - Enhanced")
        self.setGeometry(100, 100, 1200, 800)
        
        self.audio_data = None
        self.sample_rate = None
        self.file_path = None
        self.labels = []
        self.current_label_index = -1
        
        # ...其他初始化代码...
        self.current_file_index = 0
        self.wav_files = []

        self.init_ui()
        self.init_menubar()
        
        # 音频播放相关
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback)
        self.is_playing = False
        self.playback_start_pos = 0
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 控制区域
        control_layout = QHBoxLayout()

        # 添加文件夹打开按钮
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)

        self.open_btn = QPushButton("Open Audio")
        self.open_btn.clicked.connect(self.open_audio)

        # 添加上一首/下一首按钮
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_file)
        self.next_btn.clicked.connect(self.next_file)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        self.play_btn.setShortcut('Space')
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        
        self.zoom_in_btn = QPushButton("Zoom In (X)")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_in(dir="x"))

        self.zoom_out_btn = QPushButton("Zoom Out (X)")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_out("x"))

        self.zoom_in_btn_y = QPushButton("Zoom In(Y)(W)")
        self.zoom_in_btn_y.clicked.connect(lambda: self.zoom_in(dir="y"))
        self.zoom_in_btn_y.setShortcut('W')

        self.zoom_out_btn_y = QPushButton("Zoom Out(Y)(S)")
        self.zoom_out_btn_y.clicked.connect(lambda: self.zoom_out("y"))
        self.zoom_out_btn_y.setShortcut('S')

        self.add_label_btn = QPushButton("Add Label (ctrl+A)")
        self.add_label_btn.clicked.connect(self.add_label)
        self.add_label_btn.setEnabled(False)
        self.add_label_btn.setShortcut('Ctrl+A')

        self.edit_label_btn = QPushButton("Edit Label(ctrl+e)")
        self.edit_label_btn.clicked.connect(self.edit_label)
        self.edit_label_btn.setEnabled(False)
        self.edit_label_btn.setShortcut('Ctrl+E')
        
        self.delete_label_btn = QPushButton("Delete Label(ctrl+d)")
        self.delete_label_btn.clicked.connect(self.delete_label)
        self.delete_label_btn.setEnabled(False)
        self.delete_label_btn.setShortcut('Ctrl+D')
        
        self.save_btn = QPushButton("Save Labels(ctrl+s)")
        self.save_btn.clicked.connect(self.save_labels)
        self.save_btn.setEnabled(False)
        self.save_btn.setShortcut('Ctrl+S')

        # 添加到控制栏
        control_layout.addWidget(self.open_folder_btn)
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.zoom_in_btn_y)
        control_layout.addWidget(self.zoom_out_btn_y)

        control_layout.addWidget(self.add_label_btn)
        control_layout.addWidget(self.edit_label_btn)
        control_layout.addWidget(self.delete_label_btn)
        control_layout.addWidget(self.save_btn)

        # 添加到主布局
        main_layout.addLayout(control_layout)
        
        # 分割视图和标签列表
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 上部：音频视图
        view_widget = QWidget()
        view_layout = QVBoxLayout()
        view_widget.setLayout(view_layout)
        
        # 波形显示
        self.waveform_view = WaveformViewer()
        
        # # # 频谱图显示
        self.spectrogram_view = SpectrogramViewer()
    
        # # 同步视图缩放
        self.waveform_view.linkView(self.spectrogram_view)
        self.spectrogram_view.linkView(self.waveform_view)

        # 连接选择信号
        self.waveform_view.selection_changed.connect(self.on_selection_changed)

        view_layout.addWidget(self.waveform_view)
        view_layout.addWidget(self.spectrogram_view)

        # 下部：标签列表
        self.label_list = QListWidget()
        self.label_list.itemDoubleClicked.connect(self.on_label_double_clicked)

        splitter.addWidget(view_widget)
        splitter.addWidget(self.label_list)
        splitter.setSizes([600, 200])

        # 时间轴滑块
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.sliderMoved.connect(self.slider_moved)
        
        main_layout.addWidget(self.time_slider)

        main_layout.addWidget(QStatusBar())

    def init_menubar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("File")
        
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_audio)
        open_action.setShortcut('Ctrl+O')
        
        open_d_action = file_menu.addAction("Open Folder")
        open_d_action.triggered.connect(self.open_folder)
        open_d_action.setShortcut('Ctrl+Shift+O')

        save_action = file_menu.addAction("Save Labels")
        save_action.triggered.connect(self.save_labels)

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("Edit")
        
        add_label_action = edit_menu.addAction("Add Label")
        add_label_action.triggered.connect(self.add_label)
        
        edit_label_action = edit_menu.addAction("Edit Label")
        edit_label_action.triggered.connect(self.edit_label)
        
        delete_label_action = edit_menu.addAction("Delete Label")
        delete_label_action.triggered.connect(self.delete_label)
        
        clear_labels_action = edit_menu.addAction("Clear Labels")
        clear_labels_action.triggered.connect(self.clear_labels)
    
    def open_folder(self):
        """打开文件夹并加载所有WAV文件"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with WAV Files")
        if folder_path:
            self.wav_files = sorted(find_wav_files(folder_path))
            if self.wav_files:
                self.current_file_index = 0
                self.load_audio_file(self.wav_files[0])
                self.update_nav_buttons()

    def load_audio_file(self, file_path):
        """加载单个音频文件"""
        try:
            self.audio_data, self.sample_rate = sf.read(file_path)
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0]
            self.file_path = file_path
            self.display_audio()
            self.play_btn.setEnabled(True)
            self.add_label_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")

    def prev_file(self):
        """加载下一首，自动保存当前文件的标注"""
        if self.labels:  # 如果有未保存的标注
            self.save_labels_auto()
        self.clear_labels()

        """加载上一首"""
        if self.wav_files:
            if self.current_file_index > 0:
                self.current_file_index -= 1
            else:
                self.current_file_index = len(self.wav_files) - 1
            print("[AV_Labeller] ", __class__, " prev_file: ", self.current_file_index)
            self.load_audio_file(self.wav_files[self.current_file_index])
            self.load_labels_auto()
            self.update_nav_buttons()

    def next_file(self):
        """加载下一首，自动保存当前文件的标注"""
        if self.labels:  # 如果有未保存的标注
            self.save_labels_auto()

        self.clear_labels()

        if self.wav_files:
            if self.current_file_index < len(self.wav_files) - 1:
                self.current_file_index += 1
            else:
                self.current_file_index = 0

            print("[AV_Labeller] ", __class__, " next_file: ", self.current_file_index)

            self.load_audio_file(self.wav_files[self.current_file_index])
            self.load_labels_auto()
            self.update_nav_buttons()

    def save_labels_auto(self):
        """自动保存标注（与音频文件同名但扩展名为.json）"""
        if not self.file_path: return 
        if not self.labels:
            self.statusBar().showMessage("No audio file loaded, No labels added, Pre Label will be droped")
            return

        json_path = os.path.splitext(self.file_path)[0] + '.json'
        try:
            save_data = {
                    "audio_file": self.file_path,
                    "sample_rate": self.sample_rate,
                    "duration/s": len(self.audio_data) / self.sample_rate,
                    "labels": self.labels
                }
            with open(json_path, 'w') as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            print(f"Failed to auto-save labels: {str(e)}")

    def load_labels_auto(self):
        """自动加载标注"""
        if not self.file_path:
            self.statusBar().showMessage("No audio file loaded")
            return
        
        json_path = os.path.splitext(self.file_path)[0] + '.json'
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.labels = data.get('labels', [])
                    self.display_labels()
            except Exception as e:
                self.statusBar().showMessage(f"Failed to load labels: {str(e)}")
                print(f"Failed to load labels: {str(e)}")
        else:
            self.clear_labels()
        
    def update_nav_buttons(self):
        """更新导航按钮状态"""
        self.prev_btn.setEnabled(len(self.wav_files) > 0) #  and self.current_file_index > 0)
        self.next_btn.setEnabled(len(self.wav_files) > 0) #  and self.current_file_index < len(self.wav_files) - 1)
        # 更新窗口标题显示当前文件位置
        if self.wav_files:
            self.setWindowTitle(f"PyAudioLabeler - {self.current_file_index + 1}/{len(self.wav_files)}: {os.path.basename(self.file_path)}")

    def display_labels(self):
        # 清除之前的标签显示
        for item in self.waveform_view.allChildItems():
            if isinstance(item, pg.LinearRegionItem):
                self.waveform_view.removeItem(item)
        # 添加新的标签区域
        for index, label in enumerate( self.labels ):
            start, end, label = label["start"], label["end"], label["label"]

            region = pg.LinearRegionItem(values=[start, end])

            # # 在波形上显示标签区域
            region.setBrush(QBrush(QColor(0, 255, 0, 50)))
            region.setZValue(5)
            self.waveform_view.addItem(region)

            # # 添加到列表控件
            item = QListWidgetItem(f"{start:.6f}-{end:.6f}: {label}")
            item.setData(Qt.UserRole, index)
            self.label_list.addItem(item)
            
            # # 更新按钮状态
            self.edit_label_btn.setEnabled(True)
            self.delete_label_btn.setEnabled(True)

    def open_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*)"
        )
        
        if file_path:
            self.file_path = file_path
            try:
                # 使用soundfile读取音频数据
                self.audio_data, self.sample_rate = sf.read(file_path)

                # 如果是立体声，取左声道
                if len(self.audio_data.shape) > 1:
                    self.audio_data = self.audio_data[:, 0]
                
                self.display_audio()
                self.play_btn.setEnabled(True)
                self.add_label_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

                # 清除旧标签
                self.clear_labels()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")
    
    def display_audio(self):
        # 计算时间轴
        duration = len(self.audio_data) / self.sample_rate
        time_axis = np.linspace(0, duration, len(self.audio_data))

        # 更新波形显示
        self.waveform_view.set_waveform(time_axis, self.audio_data)
        
        # 计算并显示频谱图
        self.display_spectrogram()
        
        # 重置缩放
        self.waveform_view.autoRange()
        self.spectrogram_view.autoRange()
        
        # 更新时间滑块范围
        self.time_slider.setRange(0, int(duration * 1000))

    def display_spectrogram(self):
        """使用 NumPy 计算稳定的频谱图"""
        # STFT参数
        n_fft = (self.sample_rate // 1000) * 4
        hop_length = n_fft // 2

        # 设置显示范围
        duration = len( self.audio_data ) / self.sample_rate
        max_freq = self.sample_rate / 2
        extent = (0, duration, 0, max_freq)

        if 1:
            stft = librosa.stft(self.audio_data.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
            spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        else:
            audio_mono = self.audio_data.astype(np.float32)
            # audio_mono = audio_mono / np.max(np.abs(audio_mono))  # 归一化到[-1,1]

            # 使用NumPy手动计算STFT
            window = np.hanning(n_fft)
            stft = np.array([
                np.fft.rfft(window * audio_mono[i:i+n_fft], n=n_fft)
                for i in range(0, len(audio_mono)-n_fft, hop_length)
            ]).T  # 转置得到 (频率bins, 时间帧)

            # 幅度转dB（数值稳定实现）
            magnitude = np.abs(stft)
            spectrogram = 20 * np.log10(
                np.maximum(magnitude, 1e-12) / np.max(magnitude)  # 避免log(0)
            )
            print(f"Spectrogram shape: {spectrogram.shape},"
                f"Range: [{np.nanmin(spectrogram):.1f}, {np.nanmax(spectrogram):.1f}] dB")
            
        spectrogram -= spectrogram.mean()

        self.spectrogram_view.set_spectrogram(spectrogram - np.nanmin(spectrogram), extent)

    def play_audio(self):
        if self.audio_data is None:
            return
            
        # 获取当前选择区域
        selection = self.waveform_view.get_selection()
        
        if selection:
            start, end = selection
            self.playback_start_pos = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            playback_duration = (end_sample - self.playback_start_pos) / self.sample_rate
        else:
            self.playback_start_pos = 0
            playback_duration = len(self.audio_data) / self.sample_rate
            
        # 设置播放位置
        self.time_slider.setValue(int(self.playback_start_pos / self.sample_rate * 1000))
        
        # 开始播放
        self.is_playing = True
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.playback_timer.start(50)  # 20fps更新
    
    def stop_audio(self):
        self.is_playing = False
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.waveform_view.clear_playback_pos()
    
    def update_playback(self):
        if not self.is_playing:
            return
            
        # 更新播放位置
        current_time = self.time_slider.value() / 1000.0
        new_pos = int(current_time * self.sample_rate) + int(0.05 * self.sample_rate)
        
        # 检查是否到达结束位置
        selection = self.waveform_view.get_selection()
        if selection:
            _, end = selection
            if new_pos / self.sample_rate >= end:
                self.stop_audio()
                return
        
        if new_pos >= len(self.audio_data):
            self.stop_audio()
            return
            
        # 更新滑块位置
        self.time_slider.setValue(int(new_pos / self.sample_rate * 1000))
        
        # 更新播放位置指示器
        self.waveform_view.set_playback_pos(new_pos / self.sample_rate)
    
    def slider_moved(self, value):
        # 滑块移动时更新播放位置
        time_pos = value / 1000.0
        self.waveform_view.set_playback_pos(time_pos)
        
        # 如果在播放中，更新播放位置
        if self.is_playing:
            self.playback_start_pos = int(time_pos * self.sample_rate)
    
    def zoom_in(self, dir="x"):
        if dir == "x":
            scale = (0.5, 1)
        else:
            scale = (1, 0.5)
        self.waveform_view.getViewBox().scaleBy( scale )
        self.spectrogram_view.getViewBox().scaleBy( scale )

    def zoom_out(self,  dir="x"):
        if dir == "x":
            scale = (2, 1)
        else:
            scale = (1, 2)
        self.waveform_view.getViewBox().scaleBy( scale )
        self.spectrogram_view.getViewBox().scaleBy( scale )
    
    def on_selection_changed(self, start, end):
        """当选择区域改变时更新按钮状态"""
        has_selection = start != end
        self.add_label_btn.setEnabled(has_selection)
        self.edit_label_btn.setEnabled(has_selection)
    
    def add_label(self):
        selection = self.waveform_view.get_selection()
        if not selection:
            QMessageBox.warning(self, "Warning", "Please select a region first")
            return
            
        start, end = selection
        label, ok = QInputDialog.getText(self, "Add Label", "Enter label for selected region:")
        if ok and label:
            # 添加到标签列表
            self.labels.append({
                "start": start,
                "end": end,
                "label": label
            })
            
            # 在波形上显示标签区域
            index = self.waveform_view.add_label_region(start, end, label)

            # 添加到列表控件
            item = QListWidgetItem(f"{start:.6f}-{end:.6f}: {label}")
            item.setData(Qt.UserRole, index)
            self.label_list.addItem(item)
            
            # 更新按钮状态
            self.edit_label_btn.setEnabled(True)
            self.delete_label_btn.setEnabled(True)

    def edit_label(self):
        # 获取当前选择的标签（从列表或波形选择）
        if self.label_list.currentItem():
            item = self.label_list.currentItem()
            index = item.data(Qt.UserRole)
        else:
            selection = self.waveform_view.get_selection()
            if not selection:
                QMessageBox.warning(self, "Warning", "Please select a label to edit")
                return
                
            # 查找匹配的标签
            start, end = selection
            index = self.find_label_index(start, end)
            if index is None:
                QMessageBox.warning(self, "Warning", "No label found in selected region")
                return
        
        if index >= len(self.labels):
            return
            
        label_data = self.labels[index]
        
        # 编辑对话框
        new_label, ok = QInputDialog.getText(
            self, "Edit Label", "Edit label text:", 
            text=label_data["label"]
        )
        
        if ok and new_label:
            # 更新标签
            self.labels[index]["label"] = new_label
            self.waveform_view.update_label_region(index, text=new_label)

            # 更新列表项
            for i in range(self.label_list.count()):
                item = self.label_list.item(i)
                if item.data(Qt.UserRole) == index:
                    item.setText(f"{label_data['start']:.6f}-{label_data['end']:.6f}: {new_label}")
                    break

    def delete_label(self):
        # 获取当前选择的标签

        # for item in self.waveform_view.allChildItems():
        #     if isinstance(item, pg.LinearRegionItem):
        #         self.waveform_view.removeItem(item)

        if self.label_list.currentItem():
            item = self.label_list.currentItem()
            index = item.data(Qt.UserRole)
            self.label_list.takeItem(self.label_list.row(item))
        else:
            selection = self.waveform_view.get_selection()
            if not selection:
                QMessageBox.warning(self, "Warning", "Please select a label to delete")
                return

            # 查找匹配的标签
            start, end = selection
            index = self.find_label_index(start, end)
            if index is None:
                QMessageBox.warning(self, "Warning", "No label found in selected region")
                return
        
        if index >= len(self.labels):
            return
            
        # 删除标签
        self.labels.pop(index)
        self.waveform_view.remove_label_region(index)
        
        # 更新列表中的索引
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            item_index = item.data(Qt.UserRole)
            if item_index > index:
                item.setData(Qt.UserRole, item_index - 1)
        
        # 更新按钮状态
        self.edit_label_btn.setEnabled(len(self.labels) > 0)
        self.delete_label_btn.setEnabled(len(self.labels) > 0)
    
    def find_label_index(self, start, end):
        """根据时间范围查找标签索引"""
        for i, label in enumerate(self.labels):
            if (abs(label["start"] - start) < 0.01 and 
                abs(label["end"] - end) < 0.01):
                return i
        return None

    def on_label_double_clicked(self, item):
        """双击标签列表项时定位到对应区域"""
        index = item.data(Qt.UserRole)
        if index < len(self.labels):
            label = self.labels[index]
            self.waveform_view.getViewBox().setRange(
                xRange=(label["start"], label["end"]),
                padding=0.1
            )
    
    def clear_labels(self):
        """清除所有标签"""

        # 清除之前的标签显示
        for item in self.waveform_view.allChildItems():
            if isinstance(item, pg.LinearRegionItem):
                self.waveform_view.removeItem(item)

        self.waveform_view.clear_label_regions()
        self.label_list.clear()
        self.labels = []
        # 删除标签
        # self.labels.pop(index)
        # self.waveform_view.remove_label_region(index)

        self.edit_label_btn.setEnabled(False)
        self.delete_label_btn.setEnabled(False)

    def save_labels(self):
        if not self.labels:
            QMessageBox.warning(self, "Warning", "No labels to save, writing to empty labels")
            return

        if not self.file_path:
            default_path = "labels.json"
        else:
            default_path = os.path.splitext(self.file_path)[0] + ".json"
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", default_path, 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                # 准备保存的数据
                save_data = {
                    "audio_file": self.file_path,
                    "sample_rate": self.sample_rate,
                    "duration": len(self.audio_data) / self.sample_rate,
                    "labels": self.labels
                }
                
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=4)
                    
                QMessageBox.information(self, "Success", "Labels saved successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QWidget {
        background-color: rgb(50, 50, 50);  /* 设置背景颜色为深灰色 */
        color: white;                       /* 设置文字颜色为白色，确保可读性 */
        font-size: 24px;                     /* 设置字体大小为14px */
    }""")

    window = AudioLabeler()
    window.show()
    sys.exit(app.exec_())
