"""OpenROADM-specific optical constraint validation."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# OpenROADM optical timing and power constraints
OPTICAL_CONSTRAINTS = {
    "OLM_TIMER1": {
        "value": 120,
        "unit": "seconds",
        "description": "OLM stabilization timer for optical power level adjustment",
        "min": 60,
        "max": 300,
    },
    "OLM_TIMER2": {
        "value": 20,
        "unit": "seconds",
        "description": "OLM measurement timer for power readings",
        "min": 10,
        "max": 60,
    },
    "span_loss_min": {
        "value": 0,
        "unit": "dB",
        "description": "Minimum acceptable span loss",
    },
    "span_loss_max": {
        "value": 28,
        "unit": "dB",
        "description": "Maximum acceptable span loss for standard reach",
    },
    "amplifier_gain_min": {
        "value": 0,
        "unit": "dB",
        "description": "Minimum amplifier gain",
    },
    "amplifier_gain_max": {
        "value": 30,
        "unit": "dB",
        "description": "Maximum amplifier gain",
    },
    "channel_power_min": {
        "value": -25,
        "unit": "dBm",
        "description": "Minimum per-channel optical power",
    },
    "channel_power_max": {
        "value": 5,
        "unit": "dBm",
        "description": "Maximum per-channel optical power",
    },
}

# Valid OpenROADM node types
VALID_NODE_TYPES = {"rdm", "xpdr", "ila", "extplug"}

# Valid OpenROADM card types
VALID_CARD_TYPES = {
    "transponder", "muxponder", "regen", "amplifier",
    "wssoam", "pluggable", "switch",
}


class OpticalConstraintValidator:
    """Validate OpenROADM-specific optical and physical constraints."""

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Run all optical constraint checks."""
        issues = []

        issues.extend(self._check_timer_values(content))
        issues.extend(self._check_power_levels(content))
        issues.extend(self._check_gain_values(content))
        issues.extend(self._check_span_loss(content))

        return len(issues) == 0, issues

    def _check_timer_values(self, content: str) -> List[str]:
        """Check OLM timer values are within valid ranges."""
        issues = []

        for timer_name in ("OLM_TIMER1", "OLM_TIMER2"):
            pattern = re.compile(
                rf"{timer_name}\s*[:=]?\s*(\d+)", re.IGNORECASE
            )
            for match in pattern.finditer(content):
                value = int(match.group(1))
                constraints = OPTICAL_CONSTRAINTS[timer_name]
                if value < constraints["min"] or value > constraints["max"]:
                    issues.append(
                        f"{timer_name} value {value} out of range "
                        f"[{constraints['min']}-{constraints['max']}] {constraints['unit']}"
                    )

        return issues

    def _check_power_levels(self, content: str) -> List[str]:
        """Check optical power levels are within valid ranges."""
        issues = []

        # Look for power-related values
        power_pattern = re.compile(
            r"(?:channel[_-]?power|optical[_-]?power|power[_-]?level)\s*[:=]?\s*(-?\d+\.?\d*)",
            re.IGNORECASE,
        )
        for match in power_pattern.finditer(content):
            value = float(match.group(1))
            min_power = OPTICAL_CONSTRAINTS["channel_power_min"]["value"]
            max_power = OPTICAL_CONSTRAINTS["channel_power_max"]["value"]
            if value < min_power or value > max_power:
                issues.append(
                    f"Channel power {value} dBm out of range [{min_power}, {max_power}] dBm"
                )

        return issues

    def _check_gain_values(self, content: str) -> List[str]:
        """Check amplifier gain values."""
        issues = []

        gain_pattern = re.compile(
            r"(?:amplifier[_-]?gain|target[_-]?gain|gain)\s*[:=]?\s*(-?\d+\.?\d*)",
            re.IGNORECASE,
        )
        for match in gain_pattern.finditer(content):
            value = float(match.group(1))
            min_gain = OPTICAL_CONSTRAINTS["amplifier_gain_min"]["value"]
            max_gain = OPTICAL_CONSTRAINTS["amplifier_gain_max"]["value"]
            if value < min_gain or value > max_gain:
                issues.append(
                    f"Amplifier gain {value} dB out of range [{min_gain}, {max_gain}] dB"
                )

        return issues

    def _check_span_loss(self, content: str) -> List[str]:
        """Check span loss values."""
        issues = []

        span_pattern = re.compile(
            r"span[_-]?loss\s*[:=]?\s*(-?\d+\.?\d*)", re.IGNORECASE
        )
        for match in span_pattern.finditer(content):
            value = float(match.group(1))
            min_loss = OPTICAL_CONSTRAINTS["span_loss_min"]["value"]
            max_loss = OPTICAL_CONSTRAINTS["span_loss_max"]["value"]
            if value < min_loss or value > max_loss:
                issues.append(
                    f"Span loss {value} dB out of range [{min_loss}, {max_loss}] dB"
                )

        return issues

    def get_constraints_info(self) -> Dict:
        """Return all known optical constraints for reference."""
        return OPTICAL_CONSTRAINTS
