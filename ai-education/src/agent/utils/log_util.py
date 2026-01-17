import os
from loguru import logger
import sys

# 用于记录日志的日志类，暂时没有把日志写入文件持久化，先用于开发时的DEBUG

root_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(root_dir, "logs")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

class MyLogger:
    def __init__(self):
        self.logger = logger
        self.logger.remove()
        self.logger.add(sys.stdout, level="DEBUG",
                        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                               "{process.name} |"
                               "{thread.name} | "
                               "<level>{level: <8}</level> | "
                               "<cyan>{module}</cyan>:<cyan>{line}</cyan> - "
                               "<level>{message}</level>")

    def get_logger(self):
        return self.logger

log = MyLogger().get_logger()

if __name__ == "__main__":
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")