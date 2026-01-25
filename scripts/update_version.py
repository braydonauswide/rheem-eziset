#!/usr/bin/env python3
"""Update integration version across this repository.

This integration tracks two distinct version concepts:

- Integration version: stored in `custom_components/rheem_eziset/manifest.json` (and mirrored in
  `custom_components/rheem_eziset/const.py` for convenience).
- Minimum supported Home Assistant version: stored in `requirements-ha.txt`, `hacs.json`, and docs.

Historically this script updated *both* with the same value, which can corrupt replacements (e.g.
`\12026...` backref ambiguity) and incorrectly bump the HA minimum when cutting a new integration
release.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def validate_version(version: str) -> bool:
    """Validate version format (semver: X.Y.Z)."""
    pattern = r"^\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def update_manifest_json(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update manifest.json version."""
    manifest_path = repo_root / "custom_components" / "rheem_eziset" / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        old_version = data.get("version", "")
        if old_version == version:
            return True  # No change needed
        if not dry_run:
            data["version"] = version
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        print(f"{'[DRY RUN] ' if dry_run else ''}Updated manifest.json: {old_version} -> {version}")
        return True
    except Exception as e:
        print(f"Error updating manifest.json: {e}", file=sys.stderr)
        return False


def update_const_py(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update const.py VERSION constant."""
    const_path = repo_root / "custom_components" / "rheem_eziset" / "const.py"
    try:
        content = const_path.read_text(encoding="utf-8")
        pattern = r'(VERSION = ")([^"]+)(")'
        match = re.search(pattern, content)
        if not match:
            print("Warning: Could not find VERSION constant in const.py", file=sys.stderr)
            return False
        old_version = match.group(2)
        if old_version == version:
            return True  # No change needed
        if not dry_run:
            # Use \g<> to avoid backref ambiguity when version starts with digits (e.g. 2026.x.y).
            new_content = re.sub(pattern, rf"\g<1>{version}\g<3>", content)
            const_path.write_text(new_content, encoding="utf-8")
        print(f"{'[DRY RUN] ' if dry_run else ''}Updated const.py: {old_version} -> {version}")
        return True
    except Exception as e:
        print(f"Error updating const.py: {e}", file=sys.stderr)
        return False


def update_requirements_ha(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update requirements-ha.txt Home Assistant version (minimum supported HA)."""
    req_path = repo_root / "requirements-ha.txt"
    try:
        content = req_path.read_text(encoding="utf-8")
        pattern = r"(homeassistant==)(\d+\.\d+\.\d+)"
        match = re.search(pattern, content)
        if not match:
            print("Warning: Could not find homeassistant version in requirements-ha.txt", file=sys.stderr)
            return False
        old_version = match.group(2)
        if old_version == version:
            return True  # No change needed
        if not dry_run:
            new_content = re.sub(pattern, rf"\g<1>{version}", content)
            req_path.write_text(new_content, encoding="utf-8")
        print(f"{'[DRY RUN] ' if dry_run else ''}Updated requirements-ha.txt: {old_version} -> {version}")
        return True
    except Exception as e:
        print(f"Error updating requirements-ha.txt: {e}", file=sys.stderr)
        return False


def update_hacs_json(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update hacs.json Home Assistant version (minimum supported HA)."""
    hacs_path = repo_root / "hacs.json"
    try:
        with hacs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        old_version = data.get("homeassistant", "")
        if old_version == version:
            return True  # No change needed
        if not dry_run:
            data["homeassistant"] = version
            with hacs_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        print(f"{'[DRY RUN] ' if dry_run else ''}Updated hacs.json: {old_version} -> {version}")
        return True
    except Exception as e:
        print(f"Error updating hacs.json: {e}", file=sys.stderr)
        return False


def update_readme(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update README.md Home Assistant minimum version reference."""
    readme_path = repo_root / "README.md"
    try:
        content = readme_path.read_text(encoding="utf-8")
        pattern = r'(Targets \*\*Home Assistant ≥ )(\d+\.\d+\.\d+)(\*\*)'
        match = re.search(pattern, content)
        if not match:
            print("Warning: Could not find version reference in README.md", file=sys.stderr)
            return False
        old_version = match.group(2)
        if old_version == version:
            return True  # No change needed
        if not dry_run:
            new_content = re.sub(pattern, rf"\g<1>{version}\g<3>", content)
            readme_path.write_text(new_content, encoding="utf-8")
        print(f"{'[DRY RUN] ' if dry_run else ''}Updated README.md: {old_version} -> {version}")
        return True
    except Exception as e:
        print(f"Error updating README.md: {e}", file=sys.stderr)
        return False


def update_agents_docs(version: str, repo_root: Path, dry_run: bool = False) -> bool:
    """Update agents docs Home Assistant minimum version reference if present."""
    doc_paths = [
        repo_root / "agents.md",
        repo_root / "AGENTS.md",
    ]
    # Pattern for "targets HA ≥ VERSION"
    pattern = r"(targets HA ≥ )(\d+\.\d+\.\d+)"

    success = True
    for path in doc_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
            match = re.search(pattern, content)
            if not match:
                continue  # Nothing to update in this file

            old_version = match.group(2)
            if old_version == version:
                continue

            if not dry_run:
                new_content = re.sub(pattern, rf"\g<1>{version}", content)
                path.write_text(new_content, encoding="utf-8")

            print(
                f"{'[DRY RUN] ' if dry_run else ''}Updated {path.name}: {old_version} -> {version}"
            )
        except Exception as e:
            print(f"Error updating {path.name}: {e}", file=sys.stderr)
            success = False
    return success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update version across all files in the repository")
    parser.add_argument(
        "integration_version",
        nargs="?",
        help="Integration version to set (e.g., 1.0.1)",
    )
    parser.add_argument(
        "--ha-min-version",
        dest="ha_min_version",
        help="Minimum supported Home Assistant version (e.g., 2026.1.4)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    args = parser.parse_args()
    
    import os

    # Integration version from argument or environment variable (back-compat: VERSION).
    integration_version = (
        args.integration_version
        or os.environ.get("INTEGRATION_VERSION")
        or os.environ.get("VERSION")
        or ""
    ).lstrip("v")

    if not integration_version:
        print("Error: integration version not provided. Use --help for usage.", file=sys.stderr)
        return 1

    if not validate_version(integration_version):
        print(
            f"Error: Invalid integration version format: {integration_version}. Expected format: X.Y.Z (e.g., 1.0.1)",
            file=sys.stderr,
        )
        return 1

    # Optional HA minimum version (separate from integration version)
    ha_min_version = (args.ha_min_version or os.environ.get("HA_MIN_VERSION") or "").lstrip("v")
    if ha_min_version and not validate_version(ha_min_version):
        print(
            f"Error: Invalid HA minimum version format: {ha_min_version}. Expected format: X.Y.Z (e.g., 2026.1.4)",
            file=sys.stderr,
        )
        return 1
    
    # Find repository root (assume script is in scripts/ directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    if not (repo_root / "custom_components" / "rheem_eziset" / "manifest.json").exists():
        print(f"Error: Could not find repository root. Expected manifest.json at {repo_root / 'custom_components' / 'rheem_eziset' / 'manifest.json'}", file=sys.stderr)
        return 1
    
    # Update all files
    success = True
    success &= update_manifest_json(integration_version, repo_root, args.dry_run)
    success &= update_const_py(integration_version, repo_root, args.dry_run)

    # Only update HA-min files if explicitly requested.
    if ha_min_version:
        success &= update_requirements_ha(ha_min_version, repo_root, args.dry_run)
        success &= update_hacs_json(ha_min_version, repo_root, args.dry_run)
        success &= update_readme(ha_min_version, repo_root, args.dry_run)
        success &= update_agents_docs(ha_min_version, repo_root, args.dry_run)
    
    if success:
        if args.dry_run:
            print("\nDry run completed successfully. No files were modified.")
        else:
            if ha_min_version:
                print(
                    f"\nIntegration version updated to {integration_version}; HA minimum updated to {ha_min_version}."
                )
            else:
                print(f"\nIntegration version updated to {integration_version}.")
        return 0
    else:
        print("\nSome errors occurred during version update.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
