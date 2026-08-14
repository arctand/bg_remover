from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QMainWindow, QMessageBox, QPushButton, QProgressBar, QStackedWidget,
    QVBoxLayout, QWidget)

from .batch import BatchProcessor
from .config import load_config
from .history import HistoryStore
from .images import discover_images
from .inference import BiRefNetBackend, detect_device
from .verification import TorchvisionPersonVerifier


class Worker(QObject):
    progress = Signal(object); finished = Signal(object); failed = Signal(str)
    def __init__(self, processor, source, output):
        super().__init__(); self.processor, self.source, self.output = processor, source, output
    @Slot()
    def run(self):
        try: self.finished.emit(self.processor.run(self.source, self.output, resume=True, callback=self.progress.emit))
        except Exception: self.failed.emit(traceback.format_exc())


class HistoryDialog(QDialog):
    def __init__(self, store: HistoryStore, parent=None):
        super().__init__(parent); self.setWindowTitle("История обработок"); self.resize(520, 430)
        layout=QVBoxLayout(self); self.list=QListWidget(); self.rows=store.load()
        for row in self.rows:
            when=row.get("timestamp","").replace("T","  "); seconds=int(row.get("duration",0))
            self.list.addItem(f"{when}\n{row.get('source_name','')} — {row.get('total',0)} фото\nГотово: {row.get('ready',0)}   На проверку: {row.get('review',0)}   Не удалось: {row.get('failed',0)}\n{seconds//60} мин {seconds%60} сек")
        layout.addWidget(self.list); button=QPushButton("Открыть результат"); button.clicked.connect(self.open_selected); layout.addWidget(button)
    def open_selected(self):
        index=self.list.currentRow()
        if 0 <= index < len(self.rows): QDesktopServices.openUrl(QUrl.fromLocalFile(self.rows[index].get("output", "")))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.cfg=load_config(); self.history=HistoryStore(); self.source_path=None; self.processor=None; self.thread=None
        self.current_output=None
        self.setWindowTitle("Background Remover"); self.setFixedSize(590, 390); self._build()

    def _build(self):
        shell=QWidget(); outer=QVBoxLayout(shell); outer.setContentsMargins(34,28,34,30); outer.setSpacing(18)
        header=QHBoxLayout(); title=QLabel("Background Remover"); title.setStyleSheet("font-size:24px;font-weight:650")
        history=QPushButton("◷  История"); history.setFlat(True); history.clicked.connect(lambda: HistoryDialog(self.history,self).exec()); header.addWidget(title); header.addStretch(); header.addWidget(history); outer.addLayout(header)
        self.stack=QStackedWidget(); outer.addWidget(self.stack,1)
        choose=QWidget(); cl=QVBoxLayout(choose); cl.setContentsMargins(0,18,0,0); cl.setSpacing(16)
        prompt=QLabel("Выберите папку с фотографиями"); prompt.setStyleSheet("font-size:17px")
        row=QHBoxLayout(); self.path_label=QLabel("Папка не выбрана"); self.path_label.setStyleSheet("padding:12px;background:#25282d;border-radius:7px"); select=QPushButton("Выбрать…"); select.clicked.connect(self._select); row.addWidget(self.path_label,1); row.addWidget(select)
        self.found=QLabel("Найдено 0 фото"); self.found.setStyleSheet("color:#aeb4bd")
        self.start=QPushButton("Начать обработку"); self.start.setMinimumHeight(48); self.start.setEnabled(False); self.start.clicked.connect(self._start); self.start.setStyleSheet("font-size:16px;font-weight:600")
        cl.addWidget(prompt); cl.addLayout(row); cl.addWidget(self.found); cl.addStretch(); cl.addWidget(self.start); self.stack.addWidget(choose)
        progress=QWidget(); pl=QVBoxLayout(progress); pl.setContentsMargins(0,32,0,12); pl.setSpacing(18)
        heading=QLabel("Обработка фотографий"); heading.setStyleSheet("font-size:20px;font-weight:600"); self.bar=QProgressBar(); self.bar.setMinimumHeight(24); self.progress_text=QLabel("0 из 0"); self.progress_text.setStyleSheet("font-size:16px"); self.stop=QPushButton("Остановить"); self.stop.clicked.connect(self._stop)
        pl.addWidget(heading); pl.addWidget(self.bar); pl.addWidget(self.progress_text); pl.addStretch(); pl.addWidget(self.stop); self.stack.addWidget(progress)
        self.setCentralWidget(shell); self.setStyleSheet("QMainWindow{background:#1b1d21;color:#f4f5f7} QLabel{color:#f4f5f7} QPushButton{padding:8px 14px} QProgressBar{text-align:center}")

    def _select(self):
        selected=QFileDialog.getExistingDirectory(self,"Выберите папку с фотографиями")
        if not selected:return
        self.source_path=Path(selected); count=sum(1 for _ in discover_images(self.source_path,self.cfg.extensions)); self.path_label.setText(str(self.source_path)); self.found.setText(f"Найдено {count:,} фото".replace(","," ")); self.start.setEnabled(count>0)

    def _start(self):
        if not self.source_path:return
        info=detect_device(self.cfg.model.precision)
        if not info.available:
            if QMessageBox.question(self,"GPU-ускорение недоступно","CUDA недоступна. Запустить на CPU?") != QMessageBox.Yes:return
        output=self.source_path.parent/f"{self.source_path.name}_result"
        self.current_output=output
        primary=BiRefNetBackend(self.cfg.model,allow_cpu=not info.available)
        verifier=TorchvisionPersonVerifier(self.cfg.qc,"cuda" if info.available else "cpu") if self.cfg.verification.enabled else None
        self.processor=BatchProcessor(self.cfg,primary,verifier=verifier); self.thread=QThread(); self.worker=Worker(self.processor,self.source_path,output); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        # Connect bound QObject methods directly. A lambda can run in the worker
        # thread in PySide, which makes showing QMessageBox from it undefined and
        # caused a several-second UI freeze after the last image.
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.stack.setCurrentIndex(1); self.stop.setEnabled(True); self.thread.start()

    @Slot(object)
    def _progress(self,p):
        self.bar.setMaximum(max(1,p.total)); self.bar.setValue(p.processed)
        if p.processed >= p.total:
            self.progress_text.setText("Завершаю обработку…")
        else:
            self.progress_text.setText(f"{p.processed:,} из {p.total:,}".replace(","," "))
    def _stop(self):
        if self.processor:self.processor.stop();self.stop.setEnabled(False);self.stop.setText("Остановка после текущего файла…")
    @Slot(object)
    def _finished(self,s):
        output=self.current_output
        self.history.add({"source_name":self.source_path.name,"source":str(self.source_path),"output":str(output),"total":s["total"],"ready":s["ready"],"review":s["review"],"failed":s["failed"],"duration":s["total_processing_time"],"stopped":s["stopped"]})
        self.stack.setCurrentIndex(0);self.stop.setText("Остановить")
        box=QMessageBox(self);box.setWindowTitle("Готово");box.setText(f"Обработано {s['total']:,} фотографий\n\nГотово: {s['ready']:,}\nНа проверку: {s['review']:,}\nНе удалось: {s['failed']:,}".replace(","," "));open_button=box.addButton("Открыть результат",QMessageBox.AcceptRole);box.addButton("Закрыть",QMessageBox.RejectRole);box.exec()
        if box.clickedButton() is open_button:QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))
    @Slot(str)
    def _failed(self,error): self.stack.setCurrentIndex(0);QMessageBox.critical(self,"Ошибка обработки",error)
    @Slot()
    def _thread_finished(self):
        self.worker=None; self.thread=None


def run_gui():
    app=QApplication.instance() or QApplication([]);app.setStyle("Fusion");window=MainWindow();window.show();return app.exec()
