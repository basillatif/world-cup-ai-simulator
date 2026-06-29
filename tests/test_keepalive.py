import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "keepalive.yml"
SCRIPT = ROOT / "scripts" / "keepalive.py"


def _parse_scalar(value: str):
    if value in {"true", "false"}:
        return value == "true"
    if value == "":
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _strip_comments_and_blanks(text: str) -> list[str]:
    return [
        line.rstrip("\n")
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _parse_yaml_subset(text: str):
    lines = _strip_comments_and_blanks(text)

    def parse_block(index: int, indent: int):
        if lines[index].lstrip().startswith("- "):
            values = []
            while index < len(lines):
                line = lines[index]
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent != indent or not line.lstrip().startswith("- "):
                    break
                item = line.lstrip()[2:]
                if ": " in item or item.endswith(":"):
                    key, _, raw_value = item.partition(":")
                    value = _parse_scalar(raw_value.strip())
                    node = {key: value}
                    index += 1
                    if index < len(lines):
                        next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                        if next_indent > indent:
                            child, index = parse_block(index, next_indent)
                            if isinstance(child, dict):
                                node.update(child)
                    values.append(node)
                else:
                    values.append(_parse_scalar(item))
                    index += 1
            return values, index

        values = {}
        while index < len(lines):
            line = lines[index]
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent != indent or line.lstrip().startswith("- "):
                break
            key, _, raw_value = line.strip().partition(":")
            raw_value = raw_value.strip()
            index += 1
            if raw_value == "|":
                block_lines = []
                while index < len(lines):
                    next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                    if next_indent <= indent:
                        break
                    block_lines.append(lines[index][next_indent:])
                    index += 1
                values[key] = "\n".join(block_lines)
            elif raw_value:
                values[key] = _parse_scalar(raw_value)
            elif index < len(lines):
                next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                if next_indent > indent:
                    values[key], index = parse_block(index, next_indent)
                else:
                    values[key] = None
            else:
                values[key] = None
        return values, index

    parsed, index = parse_block(0, 0)
    assert index == len(lines)
    return parsed


def _load_workflow():
    text = WORKFLOW.read_text()
    try:
        import yaml
    except ModuleNotFoundError:
        return _parse_yaml_subset(text)

    data = yaml.safe_load(text)
    if True in data:
        data["on"] = data.pop(True)
    return data


def _hours_from_field(field: str) -> list[int]:
    if field == "*":
        return list(range(24))
    if field.startswith("*/"):
        step = int(field[2:])
        return list(range(0, 24, step))
    hours = set()
    for part in field.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            hours.update(range(start, end + 1))
        else:
            hours.add(int(part))
    return sorted(hours)


def test_keepalive_workflow_yaml_parses():
    workflow = _load_workflow()

    assert workflow["name"] == "Keep Streamlit Awake"
    assert workflow["jobs"]["keepalive"]["runs-on"] == "ubuntu-latest"


def test_keepalive_schedule_fires_at_least_every_6h():
    workflow = _load_workflow()
    schedules = workflow["on"]["schedule"]
    assert schedules

    cron = schedules[0]["cron"]
    minute, hour, day_of_month, month, day_of_week = cron.split()
    assert day_of_month == month == day_of_week == "*"
    assert minute.isdigit()

    hours = _hours_from_field(hour)
    gaps = [
        (hours[(index + 1) % len(hours)] - current) % 24 or 24
        for index, current in enumerate(hours)
    ]
    assert max(gaps) <= 6


def test_keepalive_script_compiles():
    ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
