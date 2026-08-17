"""Explicit, reproducible demonstration faults."""

from __future__ import annotations


class FaultInjector:
    def __init__(self, mode: str | None = None) -> None:
        if mode not in {None, "fred:corrupt_once"}:
            raise ValueError(f"unsupported fault mode: {mode}")
        self.mode = mode
        self._used = False

    def apply(self, source: str, payload: bytes) -> bytes:
        if self.mode != "fred:corrupt_once" or source != "fred" or self._used:
            return payload
        text = payload.decode("utf-8-sig")
        first_line, separator, remainder = text.partition("\n")
        columns = first_line.rstrip("\r").split(",")
        if "CPIAUCNS" not in columns:
            raise ValueError("canonical FRED fixture lacks CPIAUCNS")
        columns[columns.index("CPIAUCNS")] = "BROKEN_CPI"
        line_ending = "\r\n" if first_line.endswith("\r") else "\n"
        self._used = True
        return (",".join(columns) + (line_ending if separator else "") + remainder).encode("utf-8")
