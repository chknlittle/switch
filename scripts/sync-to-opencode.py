#!/usr/bin/env python3
"""Sync Claude Code skills to OpenCode format.

Claude Code: skillname.md (flat files with YAML frontmatter)
OpenCode: skillname/SKILL.md (folder per skill)

This script reads all .md files from the source directory and creates
the OpenCode folder structure in the target directory.

Usage:
    python sync-to-opencode.py [--source DIR] [--target DIR]

Defaults:
    --source: ~/switch/skills/
    --target: ~/.config/opencode/skill/
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown content.

    Returns:
        Tuple of (frontmatter dict, body string).
    """
    if not content.startswith('---'):
        return {}, content

    # Find the closing ---
    end_match = re.search(r'\n---\n', content[3:])
    if not end_match:
        return {}, content

    frontmatter_text = content[4:end_match.start() + 3]
    body = content[end_match.end() + 4:]

    # Simple YAML parsing (just key: value pairs)
    frontmatter = {}
    for line in frontmatter_text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


def validate_opencode_name(name: str) -> bool:
    """Check if name matches OpenCode requirements.

    Must be lowercase alphanumeric with single hyphens, no leading/trailing hyphens.
    """
    return bool(re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name))


def sync_skills(source_dir: Path, target_dir: Path, dry_run: bool = False) -> None:
    """Sync skills from Claude Code format to OpenCode format.

    Args:
        source_dir: Directory containing Claude Code .md skill files.
        target_dir: Directory to create OpenCode skill folders in.
        dry_run: If True, print what would be done without making changes.
    """
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    # Find all .md files (excluding this script and special files)
    skill_files = [
        f for f in source_dir.glob('*.md')
        if f.name not in ('.gitkeep', 'README.md')
    ]

    if not skill_files:
        print(f"No skill files found in {source_dir}")
        return

    print(f"Found {len(skill_files)} skill(s) in {source_dir}")

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    synced = 0
    skipped = 0

    for skill_file in skill_files:
        content = skill_file.read_text()
        frontmatter, body = parse_frontmatter(content)

        name = frontmatter.get('name', skill_file.stem)
        description = frontmatter.get('description', '')

        # Validate name
        if not validate_opencode_name(name):
            print(f"  SKIP {skill_file.name}: name '{name}' doesn't match OpenCode pattern")
            skipped += 1
            continue

        if not description:
            print(f"  SKIP {skill_file.name}: missing description")
            skipped += 1
            continue

        # Truncate description if needed (OpenCode max 1024 chars)
        if len(description) > 1024:
            description = description[:1021] + '...'

        # Build OpenCode SKILL.md content
        opencode_content = f"""---
name: {name}
description: {description}
---

{body.strip()}
"""

        skill_dir = target_dir / name
        skill_md = skill_dir / 'SKILL.md'

        if dry_run:
            print(f"  WOULD CREATE {skill_dir}/SKILL.md")
        else:
            skill_dir.mkdir(exist_ok=True)
            skill_md.write_text(opencode_content)
            print(f"  OK {name}/SKILL.md")

        synced += 1

    print(f"\nSynced: {synced}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description='Sync Claude Code skills to OpenCode format'
    )
    parser.add_argument(
        '--source', '-s',
        type=Path,
        default=Path('~/switch/skills/'),
        help='Source directory with Claude Code .md files'
    )
    parser.add_argument(
        '--target', '-t',
        type=Path,
        default=Path('~/.config/opencode/skill/'),
        help='Target directory for OpenCode skill folders'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()
    sync_skills(args.source, args.target, args.dry_run)


if __name__ == '__main__':
    main()
