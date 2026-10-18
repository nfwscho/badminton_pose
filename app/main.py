# Windows 데스크톱 동영상 플레이어 진입점. 프로젝트 경로를 기준으로 로그와
# UI 설정을 저장하고 실행/Qt 진단/미처리 예외를 파일과 콘솔에 기록한다.
# 직접 실행과 python -m app.main 실행을 모두 지원한다. AI 모델 로딩/추론은
# 사용자가 요청한 경우에만 별도 분석 스레드에서 수행한다.
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow


def configure_logging():
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "app.log", encoding="utf-8"), logging.StreamHandler()])

    def exception_hook(kind, value, traceback):
        logging.getLogger(__name__).critical("미처리 예외", exc_info=(kind, value, traceback))
        sys.__excepthook__(kind, value, traceback)

    sys.excepthook = exception_hook
    qInstallMessageHandler(lambda kind, context, message: logging.getLogger("qt").warning("%s", message))


def main():
    configure_logging()
    logging.info("프로그램 시작: Python %s", sys.version)
    application = QApplication(sys.argv)
    application.setApplicationName("Badminton AI Coach")
    (ROOT / "output").mkdir(exist_ok=True)
    window = MainWindow(ROOT / "output" / "settings.ini")
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
