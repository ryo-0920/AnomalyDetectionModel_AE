#!/usr/bin/env python3
"""Validate a machine-readable Codex validation report.

This script intentionally uses only the Python standard library so it can run in
fresh repositories without adding project dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


ALLOWED = {
    "overall_status": {"green", "red", "blocked", "unknown"},
    "spec_status": {"ok", "unresolved", "missing"},
    "acceptance_status": {"ok", "unmapped", "unverified", "missing"},
    "test_status": {"green", "red", "not_run", "partial"},
    "red_green_status": {"ok", "not_applicable", "not_checked", "invalid"},
    "quality_status": {"ok", "required_fixes", "warnings"},
    "code_review_severity": {"required", "in_scope_recommended", "follow_up"},
    "command_result": {"passed", "failed", "expected_failed", "skipped"},
    "command_phase": {"red", "green", "validation", "manual"},
}

REQUIRED_TOP_LEVEL = [
    "schema_version",
    "feature",
    "overall_status",
    "pr_ready",
    "spec_status",
    "acceptance_status",
    "test_status",
    "red_green_status",
    "quality_status",
    "required_fixes",
    "code_review_findings",
    "unverified_items",
    "human_decision_required",
    "commands",
    "artifacts",
]


def load_report(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"report file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from None

    if not isinstance(data, dict):
        raise ValueError("report root must be a JSON object")
    return data


def require_string(report: dict[str, Any], key: str, errors: list[str]) -> None:
    value = report.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


def require_list(report: dict[str, Any], key: str, errors: list[str]) -> list[Any]:
    value = report.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    return value


def check_allowed(report: dict[str, Any], key: str, errors: list[str]) -> None:
    value = report.get(key)
    allowed = ALLOWED[key]
    if value not in allowed:
        errors.append(f"{key} must be one of {sorted(allowed)}, got {value!r}")


def validate_commands(report: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    raw_commands = require_list(report, "commands", errors)
    commands: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_commands):
        prefix = f"commands[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue

        commands.append(raw)
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            errors.append(f"{prefix}.command must be a non-empty string")

        phase = raw.get("phase")
        if phase not in ALLOWED["command_phase"]:
            errors.append(
                f"{prefix}.phase must be one of {sorted(ALLOWED['command_phase'])}, got {phase!r}"
            )

        result = raw.get("result")
        if result not in ALLOWED["command_result"]:
            errors.append(
                f"{prefix}.result must be one of {sorted(ALLOWED['command_result'])}, got {result!r}"
            )
            continue

        exit_code = raw.get("exit_code")
        if result == "skipped":
            if exit_code is not None:
                errors.append(f"{prefix}.exit_code must be null when result is skipped")
        elif not isinstance(exit_code, int):
            errors.append(f"{prefix}.exit_code must be an integer when result is {result}")
        elif result == "passed" and exit_code != 0:
            errors.append(f"{prefix}.exit_code must be 0 when result is passed")
        elif result in {"failed", "expected_failed"} and exit_code == 0:
            errors.append(f"{prefix}.exit_code must be non-zero when result is {result}")

    return commands


def validate_code_review_findings(
    report: dict[str, Any], errors: list[str]
) -> list[dict[str, Any]]:
    raw_findings = require_list(report, "code_review_findings", errors)
    findings: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_findings):
        prefix = f"code_review_findings[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue

        findings.append(raw)
        severity = raw.get("severity")
        if severity not in ALLOWED["code_review_severity"]:
            errors.append(
                f"{prefix}.severity must be one of "
                f"{sorted(ALLOWED['code_review_severity'])}, got {severity!r}"
            )

        for key in ["category", "finding", "recommendation"]:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{key} must be a non-empty string")

        path = raw.get("path")
        if path is not None and (not isinstance(path, str) or not path.strip()):
            errors.append(f"{prefix}.path must be a non-empty string or null")

    return findings


def check_artifact_paths(
    report: dict[str, Any], repo_root: str, errors: list[str]
) -> None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        return

    for key, value in artifacts.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"artifacts.{key} must be a non-empty string or null")
            continue
        path = value if os.path.isabs(value) else os.path.join(repo_root, value)
        if not os.path.exists(path):
            errors.append(f"artifacts.{key} does not exist: {value}")


def validate_readiness(
    report: dict[str, Any],
    commands: list[dict[str, Any]],
    code_review_findings: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    expected = {
        "overall_status": "green",
        "pr_ready": True,
        "spec_status": "ok",
        "acceptance_status": "ok",
        "test_status": "green",
        "quality_status": "ok",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            errors.append(f"PR-ready report requires {key}={value!r}")

    red_green_status = report.get("red_green_status")
    if red_green_status == "ok":
        has_red_evidence = any(
            command.get("phase") == "red" and command.get("result") == "expected_failed"
            for command in commands
        )
        has_green_evidence = any(
            command.get("phase") in {"green", "validation"}
            and command.get("result") == "passed"
            for command in commands
        )
        if not has_red_evidence:
            errors.append("red_green_status=ok requires a red command with result=expected_failed")
        if not has_green_evidence:
            errors.append("red_green_status=ok requires a green or validation command with result=passed")
    elif red_green_status == "not_applicable":
        exemption = report.get("red_green_exemption")
        if not isinstance(exemption, str) or not exemption.strip():
            errors.append("red_green_status=not_applicable requires red_green_exemption")
    else:
        errors.append("PR-ready report requires red_green_status to be ok or not_applicable")

    for key in ["required_fixes", "unverified_items", "human_decision_required"]:
        values = report.get(key)
        if isinstance(values, list) and values:
            errors.append(f"PR-ready report requires {key} to be empty")

    blocking_code_review = [
        finding
        for finding in code_review_findings
        if finding.get("severity") in {"required", "in_scope_recommended"}
    ]
    if blocking_code_review:
        errors.append(
            "PR-ready report requires no code_review_findings with "
            "severity=required or severity=in_scope_recommended"
        )

    if not commands:
        errors.append("PR-ready report requires at least one command entry")

    for index, command in enumerate(commands):
        result = command.get("result")
        if result in {"failed", "skipped"}:
            errors.append(f"PR-ready report cannot include commands[{index}] result={result}")

    return errors


def validate_report(
    report: dict[str, Any],
    *,
    require_pr_ready: bool,
    check_artifacts: bool,
    repo_root: str,
) -> list[str]:
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in report:
            errors.append(f"missing required field: {key}")

    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    require_string(report, "feature", errors)
    if not isinstance(report.get("pr_ready"), bool):
        errors.append("pr_ready must be a boolean")

    for key in [
        "overall_status",
        "spec_status",
        "acceptance_status",
        "test_status",
        "red_green_status",
        "quality_status",
    ]:
        if key in report:
            check_allowed(report, key, errors)

    for key in [
        "required_fixes",
        "code_review_findings",
        "unverified_items",
        "human_decision_required",
    ]:
        require_list(report, key, errors)

    code_review_findings = validate_code_review_findings(report, errors)
    commands = validate_commands(report, errors)

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")

    if check_artifacts:
        check_artifact_paths(report, repo_root, errors)

    claims_ready = report.get("overall_status") == "green" or report.get("pr_ready") is True
    if require_pr_ready or claims_ready:
        errors.extend(validate_readiness(report, commands, code_review_findings))

    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate .codex/validation/<feature>.json"
    )
    parser.add_argument("report", help="Path to the validation report JSON file")
    parser.add_argument(
        "--require-pr-ready",
        action="store_true",
        help="Fail unless the report satisfies the PR-ready gate",
    )
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Fail when artifact paths in the report do not exist",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative artifact paths",
    )
    args = parser.parse_args(argv)

    try:
        report = load_report(args.report)
    except ValueError as exc:
        print(f"validation report invalid: {exc}", file=sys.stderr)
        return 2

    repo_root = os.path.abspath(args.repo_root)
    errors = validate_report(
        report,
        require_pr_ready=args.require_pr_ready,
        check_artifacts=args.check_artifacts,
        repo_root=repo_root,
    )

    if errors:
        print("validation report rejected:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("validation report accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
