#!/usr/bin/env python3
"""
Non-invasive File Watcher for Training Log Monitoring

Implements LogWatcher class using watchdog library to monitor training log files
without interfering with the training process.

Usage:
    from src.dashboard.monitor.log_watcher import LogWatcher

    def on_log_update(records, new_count):
        print(f"New records: {new_count}")
        # Process new records...

    watcher = LogWatcher(callback=on_log_update)
    watcher.start()
    # ... do other work ...
    watcher.stop()
"""

import os
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from src.dashboard.monitor.log_parser import LogParser


class LogWatcherHandler(FileSystemEventHandler):
    """
    Event handler for training log file modifications.

    Args:
        callback: Function to call when new log records are detected.
        parser: LogParser instance for incremental parsing.
        log_filename: Name of the log file to watch (default: "training.log").
    """

    def __init__(
        self,
        callback: Callable[[Dict[str, List[Any]], int], None],
        parser: LogParser,
        log_filename: str = "training.log",
    ):
        self.callback = callback
        self.parser = parser
        self.log_filename = log_filename

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if (
            not event.is_directory
            and os.path.basename(event.src_path) == self.log_filename
        ):
            try:
                records, new_count = self.parser.parse_incremental(event.src_path)
                if new_count > 0:
                    self.callback(records, new_count)
            except Exception as e:
                print(f"Error processing log file {event.src_path}: {e}")


class LogWatcher:
    """
    Non-invasive file watcher for training log monitoring.

    Monitors the outputs/train/ directory for training.log files and
    triggers callbacks when new log entries are detected.

    Args:
        callback: Function to call when new log records are detected.
                  Signature: callback(records: Dict[str, List[Any]], new_count: int) -> None
        log_dir: Directory to monitor for log files (default: "outputs/train").
        log_filename: Name of the log file to watch (default: "training.log").
        state_dir: Directory to store parser state (default: same as log_dir).

    Attributes:
        observer: Watchdog observer instance.
        handler: LogWatcherHandler instance.
        parser: LogParser instance for incremental parsing.
        is_running: Boolean indicating if watcher is active.

    Examples:
        >>> def on_update(records, count):
        ...     print(f"Detected {count} new log records")
        ...
        >>> watcher = LogWatcher(callback=on_update)
        >>> watcher.start()
        >>> # Training process writes to outputs/train/pinn_*/training.log
        >>> # Callback is triggered automatically
        >>> watcher.stop()
    """

    def __init__(
        self,
        callback: Callable[[Dict[str, List[Any]], int], None],
        log_dir: str = "outputs/train",
        log_filename: str = "training.log",
        state_dir: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir).resolve()
        self.log_filename = log_filename
        self.state_dir = Path(state_dir).resolve() if state_dir else self.log_dir
        self.parser = LogParser(state_dir=str(self.state_dir))
        self.handler = LogWatcherHandler(
            callback=callback, parser=self.parser, log_filename=log_filename
        )
        self.observer = Observer()
        self.is_running = False
        self._history = deque(maxlen=1000)

    def start(self) -> None:
        """Start monitoring the log directory."""
        if not self.log_dir.exists():
            print(f"Log directory {self.log_dir} does not exist, creating...")
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.observer.schedule(self.handler, str(self.log_dir), recursive=True)
        self.observer.start()
        self.is_running = True
        print(f"Started watching {self.log_dir} for {self.log_filename} changes")

    def stop(self) -> None:
        """Stop monitoring and clean up resources."""
        if self.is_running:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            print("Stopped log watcher")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
