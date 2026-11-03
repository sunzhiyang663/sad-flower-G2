"""PyYAML + 递归 merge；不依赖 Hydra，不接触 GDK。"""
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(task=None, overrides=()):
    folder = PROJECT_ROOT / "config"
    base = yaml.safe_load((folder / "base.yaml").read_text())
    defaults = base.pop("defaults")
    config = base
    task_name = task
    for entry in defaults:
        group, name = next(iter(entry.items()))
        if group == "task":
            task_name = task_name or name
        else:
            config = merge(config, yaml.safe_load((folder / f"{name}.yaml").read_text()))
    config = merge(config, yaml.safe_load((folder / "task" / f"{task_name}.yaml").read_text()))
    apply_overrides(config, overrides)
    for key, path in config["paths"].items():
        config["paths"][key] = str((PROJECT_ROOT / Path(path).expanduser()).resolve())
    return config


def apply_overrides(config, overrides):
    """对已合并的配置应用已有键覆盖；同样供统一任务入口使用。"""
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"覆盖参数应为 key=value：{item}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"未知配置项：{key}")
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise ValueError(f"未知配置项：{key}")
        target[parts[-1]] = yaml.safe_load(value)
    return config


def required(config, path):
    value = config
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    if value is None:
        raise ValueError(f"{path} 尚未配置")
    return value
