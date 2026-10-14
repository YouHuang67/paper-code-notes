#!/usr/bin/env python3
"""Daily workflow CLI and cron management; stages live in daily_pipeline.py."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'scripts/paper-daily.json'
CRON_MARK = '# paper-daily'
CRON_LOG = ROOT / 'refs/scans/paper-daily-cron.log'


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def save_config(settings: dict) -> None:
    import daily_pipeline
    daily_pipeline.atomic(CONFIG_PATH, settings)


def parse_clock(value: str) -> tuple[int, int]:
    hour, minute = map(int, value.split(':'))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError('invalid HH:MM clock')
    return hour, minute


def cron_line(settings: dict) -> str:
    # System cron uses the machine timezone; refuse silent timezone mismatches.
    return f"* 8-12 * * 1-5 /bin/bash {shlex.quote(str(ROOT / 'scripts/daily_job.sh'))} >> {shlex.quote(str(CRON_LOG))} 2>&1 {CRON_MARK}"


def sync_cron(settings: dict) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if datetime.now().astimezone().utcoffset() != datetime.now(ZoneInfo(settings['timezone'])).utcoffset():
        raise RuntimeError('cron host timezone must match Asia/Shanghai')
    current = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if current.returncode not in (0, 1):
        raise RuntimeError('cannot read crontab')
    CRON_LOG.parent.mkdir(parents=True, exist_ok=True)
    kept = [line for line in current.stdout.splitlines() if CRON_MARK not in line]
    kept.append(cron_line(settings))
    subprocess.run(['crontab', '-'], input='\n'.join(kept) + '\n', text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description='Prepare, publish and deliver daily papers')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'publish', 'deliver', 'tick', 'status', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--day')
        command.add_argument('--dry-run', action='store_true')
    sub.add_parser('install')
    sub.add_parser('list')
    clock = sub.add_parser('time')
    clock.add_argument('clock')
    add = sub.add_parser('add')
    add.add_argument('name')
    add.add_argument('--scope', required=True)
    add.add_argument('--reject', required=True)
    remove = sub.add_parser('remove')
    remove.add_argument('name')
    args = parser.parse_args()
    settings = load_config()
    if args.command in ('prepare', 'publish', 'deliver', 'tick', 'status', 'run'):
        import daily_pipeline
        forwarded = ['tick' if args.command == 'run' else args.command]
        if args.day: forwarded += ['--day', args.day]
        if args.dry_run: forwarded.append('--dry-run')
        daily_pipeline.main(forwarded)
    elif args.command == 'install':
        sync_cron(settings)
        print(cron_line(settings))
    elif args.command == 'list':
        print(f"delivery {settings['checkTime']} {settings['timezone']}")
        for topic in settings['topics']:
            print(topic['name'], 'on' if topic.get('enabled', True) else 'off')
    elif args.command == 'time':
        parse_clock(args.clock)
        if not '08:00' <= args.clock <= '12:55':
            raise ValueError('delivery time must be within the scheduled morning window')
        settings['checkTime'] = args.clock
        save_config(settings)
        sync_cron(settings)
    elif args.command == 'add':
        if any(t['name'] == args.name for t in settings['topics']):
            raise ValueError('topic already exists')
        settings['topics'].append({'name': args.name, 'scope': args.scope, 'reject': args.reject, 'enabled': True})
        save_config(settings)
    elif args.command == 'remove':
        if not any(t['name'] == args.name for t in settings['topics']):
            raise ValueError('topic not found')
        settings['topics'] = [t for t in settings['topics'] if t['name'] != args.name]
        save_config(settings)


if __name__ == '__main__':
    main()
