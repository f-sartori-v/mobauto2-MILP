from __future__ import annotations

from pathlib import Path
import re
from typing import Optional


def _as_float(val: str | None) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def parse_cplex_log_text(text: str) -> dict[str, Optional[float] | str]:
    """Parse CPLEX log text for incumbent/bound/gap and first-incumbent time.

    Returns a dict with keys: best_integer, best_bound, gap, source, first_incumbent_time_s.
    - gap is returned as a ratio (e.g., 0.0445 for 4.45%), when available.
    - source indicates which section was used: "summary" or "node_table".
    """
    best_integer = None
    best_bound = None
    gap = None
    source = None
    first_incumbent_time_s = None

    # 1) Summary lines near the end: "Best Integer = ..." etc.
    best_int_match = None
    best_bound_match = None
    gap_match = None
    for m in re.finditer(r"Best\s+Integer\s*=\s*([-\d\.eE\+]+)", text):
        best_int_match = m
    for m in re.finditer(r"Best\s+Bound\s*=\s*([-\d\.eE\+]+)", text):
        best_bound_match = m
    for m in re.finditer(r"Gap\s*=\s*([-\d\.eE\+]+)\s*%?", text):
        gap_match = m

    if best_int_match or best_bound_match:
        best_integer = _as_float(best_int_match.group(1) if best_int_match else None)
        best_bound = _as_float(best_bound_match.group(1) if best_bound_match else None)
        if gap_match:
            gap_val = _as_float(gap_match.group(1))
            if gap_val is not None:
                gap = gap_val / 100.0
        source = "summary"

    # 2) Node table line fallback: "Node Left Objective IInf Best Integer Best Bound ItCnt Gap"
    if best_integer is None or best_bound is None:
        line_re = re.compile(
            r"^\s*\*?\s*\d+\+?\s+\d+\s+([-\d\.eE\+]+)\s+\d+\s+([-\d\.eE\+]+)\s+([-\d\.eE\+]+)\s+\d+\s+([-\d\.eE\+]+)%?",
            re.MULTILINE,
        )
        last = None
        for m in line_re.finditer(text):
            last = m
        if last:
            best_integer = _as_float(last.group(2))
            best_bound = _as_float(last.group(3))
            gap_val = _as_float(last.group(4))
            if gap_val is not None:
                gap = gap_val / 100.0
            source = "node_table"

    # 3) Best-effort parse for time to first incumbent from log phrases.
    # Examples vary by CPLEX mode/version, so keep this conservative.
    first_patterns = [
        r"Solution time\s*=\s*([-\d\.eE\+]+)\s*sec",
        r"time\s*=\s*([-\d\.eE\+]+)\s*sec\.\s*Deterministic time",
        r"Found incumbent of value [^\n]* after\s*([-\d\.eE\+]+)\s*sec",
    ]
    for pat in first_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            first_incumbent_time_s = _as_float(m.group(1))
            if first_incumbent_time_s is not None:
                break

    return {
        "best_integer": best_integer,
        "best_bound": best_bound,
        "gap": gap,
        "source": source,
        "first_incumbent_time_s": first_incumbent_time_s,
    }


def parse_cplex_log_bounds(log_path: str | Path | None) -> dict[str, Optional[float] | str]:
    """Parse a CPLEX log file to extract best integer, best bound, gap, and first-incumbent time."""
    if not log_path:
        return {"best_integer": None, "best_bound": None, "gap": None, "source": None, "first_incumbent_time_s": None}
    try:
        p = Path(log_path)
        if not p.exists():
            return {"best_integer": None, "best_bound": None, "gap": None, "source": None, "first_incumbent_time_s": None}
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"best_integer": None, "best_bound": None, "gap": None, "source": None, "first_incumbent_time_s": None}
    return parse_cplex_log_text(text)


__all__ = ["parse_cplex_log_bounds", "parse_cplex_log_text"]
