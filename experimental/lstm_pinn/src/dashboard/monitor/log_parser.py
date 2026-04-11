#!/usr/bin/env python3
"""
Non-invasive Log Parser for Dashboard Monitoring

Provides incremental parsing capability for training log files.
Reuses core parsing logic from log_parsing.py.

Usage:
    from src.dashboard.monitor.log_parser import LogParser

    parser = LogParser()
    records = parser.parse("outputs/train/pinn_*/training.log")
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from src.dashboard.monitor.log_parsing import (
    parse_training_log,
    find_log_path,
    TAG_MAP,
)


class LogParser:
    """
    Non-invasive incremental log parser for training monitoring.

    Extends scripts.analysis.training.log_parsing with:
    - Incremental parsing (track position)
    - State persistence

    Args:
        state_dir: Directory to store parse state for incremental parsing.
        tag_map: Custom tag mapping dictionary. Uses TAG_MAP by default.

    Examples:
        >>> parser = LogParser()
        >>> records = parser.parse("outputs/train/pinn_*/training.log")
        >>> parser.parse("outputs/train/pinn_*/training.log", increment=True)
    """

    STATE_FILE = ".log_parser_state.json"

    def __init__(
        self,
        state_dir: Optional[str] = None,
        tag_map: Optional[Dict[str, str]] = None,
    ):
        self.state_dir = state_dir or "."
        self.state_file = os.path.join(self.state_dir, self.STATE_FILE)
        self.tag_map = tag_map if tag_map is not None else TAG_MAP.copy()
        self.last_position: Dict[str, int] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load parse state from file if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.last_position = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.last_position = {}

    def _save_state(self) -> None:
        """Save parse state to file."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.last_position, f, indent=2)
        except IOError:
            pass

    def parse(
        self,
        log_path: str,
        increment: bool = False,
        reset: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Parse training log file.

        Delegates to scripts.analysis.training.log_parsing.parse_training_log()
        for core parsing logic.

        Args:
            log_path: Path to log file.
            increment: If True, only parse new content since last position.
            reset: If True, start parsing from beginning regardless of state.

        Returns:
            Dictionary with parsed data, each key maps to a list.
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"log file not found: {log_path}")

        # For non-incremental parsing, use the shared function directly
        if not increment:
            result = parse_training_log(log_path)
            # Update state to end of file
            abs_log_path = os.path.abspath(log_path)
            self.last_position[abs_log_path] = os.path.getsize(log_path)
            self._save_state()
            return result

        # For incremental parsing, read only new content
        abs_log_path = os.path.abspath(log_path)
        start_pos = self.last_position.get(abs_log_path, 0)

        if reset:
            start_pos = 0

        # Read new content
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(start_pos)
            new_content = f.read()

        # Create temporary file-like parsing
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name

        try:
            result = parse_training_log(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Update state
        self.last_position[abs_log_path] = os.path.getsize(log_path)
        self._save_state()

        return result

    def parse_incremental(self, log_path: str) -> Tuple[Dict[str, List[Any]], int]:
        """
        Parse only new content since last position.

        Args:
            log_path: Path to log file.

        Returns:
            Tuple of (records, new_count) where new_count is number of new records.
        """
        records = self.parse(log_path, increment=True)
        new_count = len(records.get("epoch", []))
        return records, new_count

    def reset_position(self, log_path: Optional[str] = None) -> None:
        """Reset parse position for file(s)."""
        if log_path is None:
            self.last_position.clear()
        else:
            abs_path = os.path.abspath(log_path)
            self.last_position.pop(abs_path, None)
        self._save_state()

    @staticmethod
    def find_log(model_dir: Optional[str] = None) -> Tuple[str, str]:
        """Find training log file. Delegates to find_log_path()."""
        return find_log_path(model_dir)


# Convenience exports
__all__ = ["LogParser", "TAG_MAP", "parse_training_log", "find_log_path"]
