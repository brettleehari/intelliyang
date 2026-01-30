"""Tests for the validation pipeline."""

import pytest

from yang_chatbot.validation.optical_constraints import OpticalConstraintValidator
from yang_chatbot.validation.yang_validator import YANGValidator


class TestYANGValidator:
    def test_basic_syntax_valid(self):
        validator = YANGValidator(config={"syntax_validation": True, "semantic_validation": False, "optical_constraints": False, "escalation_enabled": False})
        content = 'container test { leaf name { type string; } }'
        result = validator.validate(content)
        assert result.is_valid

    def test_mismatched_braces(self):
        validator = YANGValidator(config={"syntax_validation": True, "semantic_validation": False, "optical_constraints": False, "escalation_enabled": False})
        content = 'container test { leaf name { type string; }'
        result = validator.validate(content)
        assert not result.is_valid
        assert any("brace" in e.lower() for e in result.errors)

    def test_semantic_list_without_key(self):
        validator = YANGValidator(config={"syntax_validation": False, "semantic_validation": True, "optical_constraints": False, "escalation_enabled": False})
        content = 'list my-list { leaf name { type string; } }'
        result = validator.validate(content)
        assert not result.is_valid or len(result.warnings) > 0

    def test_confidence_escalation(self):
        validator = YANGValidator(config={"syntax_validation": False, "semantic_validation": False, "optical_constraints": False, "escalation_enabled": True, "confidence_threshold": 0.90})
        result = validator.validate("test content", confidence=0.85)
        assert result.escalate
        assert "confidence" in result.stages_failed

    def test_no_escalation_high_confidence(self):
        validator = YANGValidator(config={"syntax_validation": False, "semantic_validation": False, "optical_constraints": False, "escalation_enabled": True, "confidence_threshold": 0.90})
        result = validator.validate("test content", confidence=0.95)
        assert not result.escalate

    def test_full_pipeline(self):
        validator = YANGValidator(config={"syntax_validation": True, "semantic_validation": True, "optical_constraints": True, "escalation_enabled": True, "confidence_threshold": 0.90})
        content = 'container device { leaf name { type string; } }'
        result = validator.validate(content, confidence=0.95)
        assert "syntax" in result.stages_passed
        assert "confidence" in result.stages_passed


class TestOpticalConstraintValidator:
    def test_valid_timer(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("OLM_TIMER1: 120")
        assert ok
        assert len(issues) == 0

    def test_invalid_timer(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("OLM_TIMER1: 500")
        assert not ok
        assert any("OLM_TIMER1" in i for i in issues)

    def test_valid_power(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("channel_power: -10")
        assert ok

    def test_invalid_power(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("channel_power: 50")
        assert not ok

    def test_gain_out_of_range(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("amplifier_gain: 40")
        assert not ok
        assert any("gain" in i.lower() for i in issues)

    def test_span_loss_out_of_range(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("span_loss: 35")
        assert not ok

    def test_no_optical_values(self):
        validator = OpticalConstraintValidator()
        ok, issues = validator.validate("container test { leaf name { type string; } }")
        assert ok

    def test_get_constraints_info(self):
        validator = OpticalConstraintValidator()
        info = validator.get_constraints_info()
        assert "OLM_TIMER1" in info
        assert "OLM_TIMER2" in info
        assert info["OLM_TIMER1"]["value"] == 120
