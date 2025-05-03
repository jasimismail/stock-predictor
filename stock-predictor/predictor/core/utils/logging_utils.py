"""
Enhanced logging utilities module.
Configures comprehensive logging for the application with full traceback support.
"""

import os
import logging
import traceback
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler
import threading

# Global flag to track if logging has been configured
_logging_configured = False
_logging_lock = threading.Lock()

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up comprehensive logging configuration with full traceback support.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    global _logging_configured
    
    # Use a lock to prevent race conditions when multiple threads try to configure logging
    with _logging_lock:
        # Return existing logger if already configured
        logger = logging.getLogger('stock_predictor')
        if _logging_configured:
            return logger
        
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "predictions.log")
        
        logger.setLevel(log_level)
        
        # Clear existing handlers
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Enhanced formatter with full traceback support for ERROR level
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                # Only include traceback for ERROR level messages with exception info
                if record.levelno >= logging.ERROR and record.exc_info:
                    record.exc_text = ''.join(traceback.format_exception(*record.exc_info))
                return super().format(record)
        
        # Standard formatter without traceback field reference
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Add exception hook for unhandled exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        sys.excepthook = handle_exception
        
        _logging_configured = True
        logger.info(f"Logging configured. Log file: {log_file}")
        return logger

def log_exception(logger: logging.Logger, exception: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback and context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
    """
    if context:
        logger.error(f"{context}: {str(exception)}", exc_info=True)
    else:
        logger.error(str(exception), exc_info=True)

def get_logger() -> logging.Logger:
    """
    Get the configured logger or set it up if not already configured.
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger('stock_predictor')
    if not logger.handlers:
        return setup_logging()
    return logger
