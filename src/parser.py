"""
src/parser.py
负责解析 JSON 文件，提取 frame_path 字段（图片路径列表）。
"""

import json
import os
import re
from typing import Any, Dict, List, Optional


def get_frame_paths(json_path: str) -> List[str]:
    """
    解析 JSON 文件，返回 frame_path 字段中的图片路径列表。

    Args:
        json_path: JSON 文件的路径。

    Returns:
        图片路径字符串列表；若字段不存在或解析失败则返回空列表。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容列表格式（如 [{...}, ...]）和字典格式
        if isinstance(data, list):
            data = data[0] if data else {}
        frame_paths = data.get("frame_path", [])
        if not isinstance(frame_paths, list):
            return []
        return [str(p) for p in frame_paths]
    except FileNotFoundError:
        print(f"[parser] 文件未找到: {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[parser] JSON 解析失败 ({json_path}): {e}")
        return []
    except Exception as e:
        print(f"[parser] 未知错误 ({json_path}): {e}")
        return []


def extract_frame_time(filename: str) -> Optional[float]:
    """
    从帧文件名中提取时间（秒），如 time_0.50s.jpg → 0.5。

    Returns:
        浮点秒数；无法提取时返回 None。
    """
    m = re.search(r'time_([\d.]+)s', os.path.basename(filename))
    return float(m.group(1)) if m else None


def get_qa_data(json_path: str) -> List[Dict]:
    """
    返回 JSON 中的 data 字段（Q&A 条目列表）。

    Returns:
        Q&A 列表；失败时返回空列表。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0] if data else {}
        return data.get("data", [])
    except Exception as e:
        print(f"[parser] 加载 data 字段失败 ({json_path}): {e}")
        return []


def load_json(json_path: str) -> Dict[str, Any]:
    """
    加载并返回 JSON 文件的完整内容。

    Args:
        json_path: JSON 文件路径。

    Returns:
        解析后的字典；失败时返回空字典。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[parser] 加载失败 ({json_path}): {e}")
        return {}
