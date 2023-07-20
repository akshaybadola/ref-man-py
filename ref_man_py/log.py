import logging
import inspect


logger = logging.getLogger("ref-man")


def debug(message):
    f = inspect.currentframe()
    prev_func = inspect.getframeinfo(f.f_back).function
    logger.debug(f" (in {prev_func}()) {message}")


def info(message):
    f = inspect.currentframe()
    prev_func = inspect.getframeinfo(f.f_back).function
    logger.info(f" (in {prev_func}()) {message}")


def warn(message):
    f = inspect.currentframe()
    prev_func = inspect.getframeinfo(f.f_back).function
    logger.warn(f" (in {prev_func}()) {message}")


def error(message):
    f = inspect.currentframe()
    prev_func = inspect.getframeinfo(f.f_back).function
    logger.error(f" (in {prev_func}()) {message}")

