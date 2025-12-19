from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List

import typer
from rich.console import Console
from rich.table import Table

from .profiles import load_profiles, Profile
from .core import (
    Issue,
    lint_jsonl,
    fix_jsonl,
    verify_system_visibility_jsonl,
    check_dataset_on_disk,
    should_fail,
)

app = typer.Typer(add_completion=False)
console = Console()


def _print_issues(title: str, issues: List[Issue]) -> None:
    table = Table(title=title)
    table.add_column("level", style="bold")
    table.add_column("code")
    table.add_column("line", justify="right")
    table.add_column("sample_id")
    table.add_column("path")
    table.add_column("message")

    for it in issues[:2000]:
        table.add_row(it.level, it.code, str(it.line), it.sample_id, it.path, it.message)

    console.print(table)
    console.print(f"[bold]Total issues:[/bold] {len(issues)}")


def _parse_expect_size(s: str) -> Tuple[int, int]:
    # "512,512" or "512x512"
    s = s.strip().lower().replace("x", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise typer.BadParameter("expect-size must be like 512,512 or 512x512")
    return int(parts[0]), int(parts[1])


@app.command("init-profiles")
def init_profiles(
    out: str = typer.Option("profiles.yaml", help="Output path for profile template (YAML)."),
) -> None:
    """Create a YAML template profile file."""
    tmpl = {
        "profiles": {
            "my-vlm": {
                "require_system": True,
                "system_visible": True,
                "system_invisible_level": "ERROR",
                "fold_system_into_user": False,
            }
        }
    }
    Path(out).write_text(json.dumps(tmpl, indent=2) if out.endswith(".json") else _to_yaml(tmpl), encoding="utf-8")
    console.print(f"[green]Wrote profile template to:[/green] {out}")


def _to_yaml(obj) -> str:
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML required for YAML output. `pip install pyyaml`")
    return yaml.safe_dump(obj, sort_keys=False)


@app.command("validate-profiles")
def validate_profiles(
    profile_file: str = typer.Option(..., "--profile-file", help="JSON/YAML profile file."),
    format: str = typer.Option("table", "--format", help="table|json"),
) -> None:
    """Load and validate profiles (CI-friendly)."""
    profs = load_profiles(profile_file)
    if format.lower() == "json":
        console.print(json.dumps({k: vars(v) for k, v in profs.items()}, indent=2))
        raise typer.Exit(code=0)

    table = Table(title="Active profile registry (merged)")
    table.add_column("name")
    table.add_column("require_system")
    table.add_column("system_visible")
    table.add_column("fold_system_into_user")
    table.add_column("system_invisible_level")

    for name in sorted(profs.keys()):
        p = profs[name]
        table.add_row(p.name, str(p.require_system), str(p.system_visible), str(p.fold_system_into_user), p.system_invisible_level)
    console.print(f"OK: loaded {len([k for k in profs.keys() if k not in ('__meta__',)])} profile(s) from {profile_file}")
    console.print(table)


@app.command("list-profiles")
def list_profiles(
    profile_file: Optional[str] = typer.Option(None, "--profile-file", help="Optional JSON/YAML file to load/override profiles."),
) -> None:
    """List available profiles."""
    profs = load_profiles(profile_file)
    table = Table(title="mmqlint profiles")
    table.add_column("name")
    table.add_column("require_system")
    table.add_column("system_visible")
    table.add_column("fold_system_into_user")
    table.add_column("system_invisible_level")
    for name in sorted(profs.keys()):
        p = profs[name]
        table.add_row(p.name, str(p.require_system), str(p.system_visible), str(p.fold_system_into_user), p.system_invisible_level)
    console.print(table)


@app.command("check")
def check(
    data: str = typer.Argument(..., help="Path to JSONL file."),
    report: Optional[str] = typer.Option(None, "--report", help="Write issues to JSON file."),
    mode: str = typer.Option("train", "--mode", help="train | infer"),
    profile: str = typer.Option("generic", "--profile", help="Profile name (see list-profiles)."),
    profile_file: Optional[str] = typer.Option(None, "--profile-file", help="Load/override profiles from a JSON/YAML file."),
    strict_typed: bool = typer.Option(False, "--strict-typed", help="Require content to be a list of typed items (no raw strings)."),
    fix: bool = typer.Option(False, "--fix/--no-fix", help="Apply safe auto-fixes and write to --out."),
    out: Optional[str] = typer.Option(None, "--out", help="Output JSONL path (required if --fix)."),
    fix_report: Optional[str] = typer.Option(None, "--fix-report", help="Write applied fixes to JSON file."),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Print summary statistics."),
    fail_on: str = typer.Option("ERROR", "--fail-on", help="Exit non-zero if any issue is >= this level (WARN|ERROR)."),
) -> None:
    profs = load_profiles(profile_file)
    if profile not in profs:
        raise typer.BadParameter(f"Unknown profile: {profile}. Use `mmqlint list-profiles`.")
    profile_obj = profs[profile]

    target_path = data
    applied_fixes: List[Issue] = []
    if fix:
        if not out:
            raise typer.BadParameter("--out is required when --fix")
        target_path, applied_fixes = fix_jsonl(data, out, strict_typed=strict_typed)
        if fix_report:
            Path(fix_report).write_text(json.dumps([vars(x) for x in applied_fixes], indent=2), encoding="utf-8")

    issues = lint_jsonl(target_path, mode=mode, profile_obj=profile_obj, strict_typed=strict_typed)

    _print_issues(f"mmqlint: {target_path} (mode={mode}, profile={profile}, strict_typed={strict_typed})", issues)

    if report:
        Path(report).write_text(json.dumps([vars(x) for x in issues], indent=2), encoding="utf-8")

    if summary:
        _print_summary(issues)

    if should_fail(issues, fail_on):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _print_summary(issues: List[Issue]) -> None:
    # summary by level and by code (top 10)
    from collections import Counter
    by_level = Counter([i.level for i in issues])
    by_code = Counter([i.code for i in issues])
    table = Table(title="Summary by level")
    table.add_column("level")
    table.add_column("count", justify="right")
    for k in ["ERROR", "WARN", "INFO"]:
        if k in by_level:
            table.add_row(k, str(by_level[k]))
    console.print(table)

    table2 = Table(title="Top issue codes")
    table2.add_column("code")
    table2.add_column("count", justify="right")
    for code, cnt in by_code.most_common(10):
        table2.add_row(code, str(cnt))
    console.print(table2)


@app.command("verify-system")
def verify_system(
    data: str = typer.Argument(..., help="Path to JSONL file."),
    profile: str = typer.Option("generic", "--profile", help="Profile name."),
    profile_file: Optional[str] = typer.Option(None, "--profile-file", help="Load/override profiles from a JSON/YAML file."),
    render_plugin: str = typer.Option(..., "--render-plugin", help="Python file with render(messages, **kwargs)->str."),
    strict_typed: bool = typer.Option(False, "--strict-typed", help="Parse content as typed items (same semantics as check)."),
    fail_on: str = typer.Option("ERROR", "--fail-on", help="Exit non-zero if any issue is >= this level (WARN|ERROR)."),
    report: Optional[str] = typer.Option(None, "--report", help="Write issues to JSON file."),
) -> None:
    profs = load_profiles(profile_file)
    if profile not in profs:
        raise typer.BadParameter(f"Unknown profile: {profile}. Use `mmqlint list-profiles`.")
    profile_obj = profs[profile]

    issues = verify_system_visibility_jsonl(
        data,
        profile_obj=profile_obj,
        render_plugin_path=render_plugin,
        strict_typed=strict_typed,
    )
    _print_issues(f"mmqlint verify-system: {data} (profile={profile})", issues)

    if report:
        Path(report).write_text(json.dumps([vars(x) for x in issues], indent=2), encoding="utf-8")

    if should_fail(issues, fail_on):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("check-dataset")
def check_dataset(
    dataset_path: str = typer.Argument(..., help="Path to HF dataset saved with datasets.save_to_disk()."),
    expect_size: Optional[str] = typer.Option(None, "--expect-size", help="Expect image size W,H or WxH (e.g., 512,512)."),
    size_policy: str = typer.Option("any", "--size-policy", help="any | consistent (no fixed size)."),
    coord_field: Optional[str] = typer.Option(None, "--coord-field", help="(Optional) field name for coordinates dict, e.g. coordinates or ann.coords"),
    coord_keys: Optional[str] = typer.Option(None, "--coord-keys", help="(Optional) comma-separated keys within coord-field, e.g. x0,y0,x1,y1"),
    fail_on: str = typer.Option("ERROR", "--fail-on", help="Exit non-zero if any issue is >= this level (WARN|ERROR)."),
    report: Optional[str] = typer.Option(None, "--report", help="Write issues to JSON file."),
) -> None:
    exp = _parse_expect_size(expect_size) if expect_size else None
    keys = [k.strip() for k in coord_keys.split(",")] if coord_keys else None

    issues = check_dataset_on_disk(
        dataset_path,
        expect_size=exp,
        size_policy=size_policy,
        coord_field=coord_field,
        coord_keys=keys,
        fail_on_level=fail_on,
    )

    _print_issues(f"mmqlint check-dataset: {dataset_path}", issues)

    if report:
        Path(report).write_text(json.dumps([vars(x) for x in issues], indent=2), encoding="utf-8")

    if should_fail(issues, fail_on):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)
