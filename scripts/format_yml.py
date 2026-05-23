#!/usr/bin/env python3
"""
Convert YAML config files from expanded multi-line format to compact line-by-line format.

Compact format uses { key: value, ... } for mappings and [a, b, c] for
lists of scalars, similar to ~/fsimpy/configs/cache_os.yml.

Usage:
    python format_yml.py [files_or_dirs...]
    python format_yml.py --dry-run configs/

    If a directory is given, all .yml files in it are processed.
    If no arguments, processes all .yml files in configs/ directory.

Options:
    --dry-run    Print converted content without modifying files
"""

import re
import sys
import yaml
from pathlib import Path


class CompactDumper(yaml.Dumper):
    pass


def represent_dict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)


def represent_list(dumper, data):
    if not data:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    first = data[0]
    if isinstance(first, dict):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    if all(not isinstance(x, (dict, list)) for x in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)


CompactDumper.add_representer(dict, represent_dict)
CompactDumper.add_representer(list, represent_list)


TOP_LEVEL_ORDER = ['constants', 'universe', 'modules', 'stats', 'portfolio']
MODULE_KEY_ORDER = ['id', 'path', 'datapath', 'fieldpath', 'fields', 'handler']
STATS_KEY_ORDER = ['moduleId', 'vwappnl', 'delay']
PORTFOLIO_KEY_ORDER = ['moduleId', 'id', 'universe', 'data1', 'data2', 'data3', 'data4', 'data', 'mode', 'op', 'op1', 'op2', 'op3', 'idx', 'tidx', 'enddate', 'startdate', 'dump']
CONSTANTS_KEY_ORDER = ['datacache', 'startdate', 'enddate', 'booksize', 'verbose', 'mode', 'refreshdays', 'tradingtimes']


def reorder_dict(d, key_order):
    ordered = {}
    for k in key_order:
        if k in d:
            ordered[k] = d[k]
    for k in d:
        if k not in ordered:
            ordered[k] = d[k]
    return ordered


def dump_value(value):
    return yaml.dump(value, Dumper=CompactDumper, default_flow_style=False,
                     allow_unicode=True, sort_keys=False, width=10000)


def format_scalar(val):
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if val is None:
        return 'null'
    s = str(val)
    if ':' in s or '#' in s or s.startswith('{') or s.startswith('[') or s.startswith('"') or s.startswith("'"):
        return f"'{s}'"
    return s


def format_dict_inline(d):
    parts = []
    for key, val in d.items():
        if isinstance(val, dict):
            parts.append(f"{key}: {format_dict_inline(val)}")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            item_strs = [format_dict_inline(item) for item in val]
            parts.append(f"{key}: [{', '.join(item_strs)}]")
        elif isinstance(val, list):
            items = ', '.join(format_scalar(v) for v in val)
            parts.append(f"{key}: [{items}]")
        else:
            parts.append(f"{key}: {format_scalar(val)}")
    return '{' + ', '.join(parts) + '}'


def has_nested_dict_list(d):
    return any(isinstance(v, list) and v and isinstance(v[0], dict) for v in d.values())


def format_portfolio_item(item, indent=4):
    pad = ' ' * indent
    item = reorder_dict(item, PORTFOLIO_KEY_ORDER)
    parts = []
    for key, val in item.items():
        if isinstance(val, dict):
            parts.append(f"{pad}{key}: {format_dict_inline(val)}")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            item_strs = [format_dict_inline(sub) for sub in val]
            parts.append(f"{pad}{key}: [")
            for s in item_strs:
                parts.append(f"{pad}  {s},")
            parts.append(f"{pad}]")
        elif isinstance(val, list):
            items = ', '.join(format_scalar(v) for v in val)
            parts.append(f"{pad}{key}: [{items}]")
        else:
            parts.append(f"{pad}{key}: {format_scalar(val)}")
    return '\n'.join(parts)


def format_portfolio(portfolio_data):
    portfolio_data = reorder_dict(portfolio_data, PORTFOLIO_KEY_ORDER)
    parts = ["portfolio:"]
    for key, val in portfolio_data.items():
        if isinstance(val, dict):
            parts.append(f"  {key}: {format_dict_inline(val)}")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            if has_nested_dict_list(val[0]):
                parts.append(f"  {key}:")
                for item in val:
                    item = reorder_dict(item, PORTFOLIO_KEY_ORDER)
                    first = True
                    for k, v in item.items():
                        if first:
                            if isinstance(v, dict):
                                parts.append(f"  - {k}: {format_dict_inline(v)}")
                            elif isinstance(v, list) and v and isinstance(v[0], dict):
                                item_strs = [format_dict_inline(sub) for sub in v]
                                parts.append(f"  - {k}: [")
                                for s in item_strs:
                                    parts.append(f"      {s},")
                                parts.append(f"    ]")
                            elif isinstance(v, list):
                                items = ', '.join(format_scalar(x) for x in v)
                                parts.append(f"  - {k}: [{items}]")
                            else:
                                parts.append(f"  - {k}: {format_scalar(v)}")
                            first = False
                        else:
                            if isinstance(v, dict):
                                parts.append(f"    {k}: {format_dict_inline(v)}")
                            elif isinstance(v, list) and v and isinstance(v[0], dict):
                                item_strs = [format_dict_inline(sub) for sub in v]
                                parts.append(f"    {k}: [")
                                for s in item_strs:
                                    parts.append(f"      {s},")
                                parts.append(f"    ]")
                            elif isinstance(v, list):
                                items = ', '.join(format_scalar(x) for x in v)
                                parts.append(f"    {k}: [{items}]")
                            else:
                                parts.append(f"    {k}: {format_scalar(v)}")
            else:
                item_strs = [format_dict_inline(item) for item in val]
                parts.append(f"  {key}: [")
                for s in item_strs:
                    parts.append(f"    {s},")
                parts.append("  ]")
        elif isinstance(val, list):
            items = ', '.join(format_scalar(v) for v in val)
            parts.append(f"  {key}: [{items}]")
        else:
            parts.append(f"  {key}: {format_scalar(val)}")
    return '\n'.join(parts)


def convert_file(filepath, dry_run=False):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    ordered_keys = [k for k in TOP_LEVEL_ORDER if k in data]
    remaining = [k for k in data if k not in TOP_LEVEL_ORDER]
    ordered_keys.extend(remaining)

    sections = []
    for key in ordered_keys:
        value = data[key]
        if key == 'portfolio':
            sections.append(format_portfolio(value))
            continue
        if key == 'modules' and isinstance(value, list):
            value = [reorder_dict(m, MODULE_KEY_ORDER) if isinstance(m, dict) else m for m in value]
        elif key == 'stats' and isinstance(value, dict):
            value = reorder_dict(value, STATS_KEY_ORDER)
        elif key == 'constants' and isinstance(value, dict):
            value = reorder_dict(value, CONSTANTS_KEY_ORDER)
        value_str = dump_value(value).rstrip('\n')
        if isinstance(value, list) and value and isinstance(value[0], dict):
            block = f"{key}:\n{value_str}"
        elif isinstance(value, dict):
            block = f"{key}: {value_str}"
        else:
            block = f"{key}: {value_str}"
        sections.append(block)

    output = '\n\n'.join(sections) + '\n'
    output = re.sub(r"'([^']*)'", r'\1', output)

    if dry_run:
        print(f"--- {filepath} ---")
        print(output)
    else:
        with open(filepath, 'w') as f:
            f.write(output)
        print(f"Converted: {filepath}")


def main():
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    if dry_run:
        args.remove('--dry-run')

    if not args:
        args = ['configs']

    files = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            files.extend(sorted(path.glob('*.yml')))
        else:
            files.append(path)

    if not files:
        print("No .yml files found", file=sys.stderr)
        sys.exit(1)

    for f in files:
        convert_file(f, dry_run)


if __name__ == '__main__':
    main()