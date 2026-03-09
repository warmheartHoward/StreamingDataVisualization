"""
app.py — Frame Sequence Visualizer 主程序

布局：
  左列 (1/3) | Block A 配置区 (上)
             | Block B 目录交互区 (下)
  ───────────┼───────────────────────────────────────
  右列 (2/3) | Block C JSON 展示区 (上)
             | Block D 图片序列交互展示区 (下)

键盘导航：← 左方向键 / → 右方向键（通过 JS postMessage 驱动隐藏按钮）
"""

import json
import os

from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.parser import extract_frame_time, get_frame_paths, get_qa_data, load_json

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Frame Sequence Visualizer",
    page_icon="🎞️",
)

st.markdown(
    """
    <style>
    /* 缩小顶部 padding */
    .block-container { padding-top: 1.5rem; }
    /* 进度条圆角 */
    .stProgress > div > div { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State 初始化 ───────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "root_dir":        "mock_data",
    "max_display":     4,
    "main_folder":     "XXX",
    "sel_mode":        False,       # 选择模式开关
    "selected":        [],          # 已勾选的文件夹列表
    "frame_paths":     [],          # 当前 JSON 中的帧路径列表
    "display_indices": [],          # 当前窗口中显示的帧索引列表（最多 3 个）
    "json_data":       None,        # 当前加载的完整 JSON 数据
    "_toast":          None,        # 待展示的 toast 消息
    "_mf_widget":      "XXX",       # 主文件夹文本框 widget key 的存储
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── 展示挂起的 Toast（在 callback 中无法直接 st.toast）────────────────────────
if st.session_state._toast:
    st.toast(st.session_state._toast, icon="🚫")
    st.session_state._toast = None

# ════════════════════════════════════════════════════════════════════════════
# 导航逻辑（作为 on_click 回调使用，避免重复触发）
# ════════════════════════════════════════════════════════════════════════════

def nav_left() -> None:
    """左方向键：窗口向左滚动（右侧删除一张，左侧补上前一张）。"""
    idx = st.session_state.display_indices
    if not idx or idx[0] == 0:
        return
    st.session_state.display_indices = [idx[0] - 1] + idx[:-1]


def nav_right() -> None:
    """右方向键：若不足 3 张则右侧扩展；已满 3 张则整体右移。"""
    idx   = st.session_state.display_indices
    paths = st.session_state.frame_paths
    if not paths:
        return
    if not idx:
        st.session_state.display_indices = [0]
        return
    nxt = idx[-1] + 1
    if nxt >= len(paths):
        return
    if len(idx) < 3:
        st.session_state.display_indices = idx + [nxt]
    else:
        st.session_state.display_indices = idx[1:] + [nxt]


# ════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════════════════

def get_subfolders(root: str) -> list:
    """返回 root 下所有直接子文件夹（排序后）。"""
    if not root or not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def find_first_json(folder_path: str):
    """在文件夹中找到第一个 JSON 文件，返回 (data_dict, file_path)。"""
    try:
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".json"):
                fpath = os.path.join(folder_path, fname)
                return load_json(fpath), fpath
    except OSError:
        pass
    return None, None


def refresh_display() -> None:
    """当选中数目达到最大展示数时，从主文件夹加载 JSON 并重置帧索引。"""
    sel   = st.session_state.selected
    root  = st.session_state.root_dir
    max_d = st.session_state.max_display
    mf    = st.session_state.main_folder

    if len(sel) < max_d or not mf or not root:
        st.session_state.json_data       = None
        st.session_state.frame_paths     = []
        st.session_state.display_indices = []
        return

    data, jpath = find_first_json(os.path.join(root, mf))
    st.session_state.json_data = data
    if data and jpath:
        raw_paths = get_frame_paths(jpath)
        json_dir  = os.path.dirname(os.path.abspath(jpath))
        fps = [
            p if os.path.isabs(p) else os.path.join(json_dir, p)
            for p in raw_paths
        ]
        st.session_state.frame_paths     = fps
        st.session_state.display_indices = [0] if fps else []
    else:
        st.session_state.frame_paths     = []
        st.session_state.display_indices = []


# ════════════════════════════════════════════════════════════════════════════
# Widget 回调
# ════════════════════════════════════════════════════════════════════════════

def on_root_change() -> None:
    st.session_state.root_dir = st.session_state._root_widget
    st.session_state.selected = []
    refresh_display()


def on_max_change() -> None:
    new_max = st.session_state._max_widget
    st.session_state.max_display = new_max
    # 若已选数量超出新上限，自动裁剪
    if len(st.session_state.selected) > new_max:
        st.session_state.selected = st.session_state.selected[:new_max]
        refresh_display()


def on_main_folder_change() -> None:
    st.session_state.main_folder = st.session_state._mf_widget
    refresh_display()


def on_pin_click(folder: str) -> None:
    st.session_state.main_folder = folder
    st.session_state._mf_widget  = folder
    refresh_display()


def on_folder_check(folder: str) -> None:
    """文件夹复选框变化时的回调：处理勾选/取消及上限拦截。"""
    checked = st.session_state[f"chk_{folder}"]
    sel     = st.session_state.selected
    max_d   = st.session_state.max_display

    if checked:
        if len(sel) >= max_d:
            # 违规：回滚复选框状态 + 触发 toast
            st.session_state[f"chk_{folder}"] = False
            st.session_state._toast = f"超过最大展示限制（上限 {max_d} 个）"
            return
        if folder not in sel:
            sel.append(folder)
    else:
        if folder in sel:
            sel.remove(folder)

    st.session_state.selected = sel
    refresh_display()


# ════════════════════════════════════════════════════════════════════════════
# 主布局
# ════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 2], gap="large")

# ─────────────────────────── 左列 ────────────────────────────────────────────
with left_col:

    # ══ Block A: 配置区 ═══════════════════════════════════════════════════════
    st.subheader("⚙️ 配置")

    st.text_input(
        "📁 数据路径（根目录）",
        value=st.session_state.root_dir,
        key="_root_widget",
        on_change=on_root_change,
        placeholder="例: mock_data  或  D:/dataset",
    )

    st.number_input(
        "🔢 最大同时展示数目",
        min_value=1, max_value=20, step=1,
        value=st.session_state.max_display,
        key="_max_widget",
        on_change=on_max_change,
    )

    st.divider()

    # ══ Block B: 目录交互区 ═══════════════════════════════════════════════════
    st.subheader("📂 目录")

    # 主文件夹展示栏
    # 注意：通过 key 管理，可由用户键入 或 点击 📌 按钮两种途径更新
    if "_mf_widget" not in st.session_state:
        st.session_state._mf_widget = st.session_state.main_folder
    st.text_input(
        "主文件夹",
        key="_mf_widget",
        on_change=on_main_folder_change,
    )

    # 选择模式开关按钮（"开启选择模式" 即需求中的显眼触发按钮）
    n_sel  = len(st.session_state.selected)
    max_d  = st.session_state.max_display
    in_sel = st.session_state.sel_mode
    btn_lbl = "🔴 退出选择模式" if in_sel else "🟢 开启选择模式"

    if st.button(btn_lbl, use_container_width=True, key="btn_sel_toggle"):
        st.session_state.sel_mode = not in_sel
        st.rerun()

    if in_sel:
        st.info(f"✅ 选择模式已开启 — 已选 **{n_sel}** / {max_d}", icon="☑️")

    # 文件夹列表
    folders = get_subfolders(st.session_state.root_dir)

    if not folders:
        if st.session_state.root_dir:
            st.warning("⚠️ 路径无效或无子文件夹")
        else:
            st.caption("请先填写数据路径")
    else:
        st.caption(f"共 {len(folders)} 个子文件夹")
        for folder in folders:
            is_sel  = folder in st.session_state.selected
            is_main = folder == st.session_state.main_folder

            fc1, fc2, fc3 = st.columns([0.08, 0.72, 0.20])

            # 复选框（仅选择模式下可见）
            with fc1:
                if in_sel:
                    st.checkbox(
                        "",
                        value=is_sel,
                        key=f"chk_{folder}",
                        on_change=on_folder_check,
                        args=(folder,),
                        label_visibility="hidden",
                    )

            # 文件夹名称（选中高亮 + 主文件夹星标）
            with fc2:
                icon  = "📂" if is_sel else "📁"
                badge = " ⭐" if is_main else ""
                style = "color:#27AE60;font-weight:600;" if is_sel else "color:#555;"
                st.markdown(
                    f'<p style="margin:2px 0;{style}">{icon} {folder}{badge}</p>',
                    unsafe_allow_html=True,
                )

            # 📌 按钮：右键双击的等效操作 — 单击即可将该文件夹设为主文件夹
            with fc3:
                st.button("📌", key=f"pin_{folder}", help="设为主文件夹",
                          on_click=on_pin_click, args=(folder,))


# ─────────────────────────── 右列 ────────────────────────────────────────────
with right_col:

    # ══ Block C: 各选中文件夹 Q&A 展示区 ═════════════════════════════════════
    st.subheader("📋 各文件夹 Q&A 结果")

    _sel  = st.session_state.selected
    _root = st.session_state.root_dir
    _fpaths = st.session_state.frame_paths
    _idx    = st.session_state.display_indices

    # 当前最新显示帧的时间（取 display_indices 最后一个元素）
    _cur_time: Optional[float] = None
    if _idx and _fpaths:
        _cur_time = extract_frame_time(_fpaths[_idx[-1]])

    if not _sel:
        st.info("💡 请在左侧开启选择模式并勾选文件夹")
    else:
        _c_cols = st.columns(len(_sel))
        for _ci, _folder in enumerate(_sel):
            with _c_cols[_ci]:
                st.markdown(
                    f'<div style="font-weight:700;font-size:1rem;'
                    f'border-bottom:2px solid #4A90D9;padding-bottom:4px;'
                    f'margin-bottom:8px">📁 {_folder}</div>',
                    unsafe_allow_html=True,
                )
                _, _jpath = find_first_json(os.path.join(_root, _folder))
                if not _jpath:
                    st.warning("无 JSON 文件")
                    continue

                _qa_list = get_qa_data(_jpath)

                if _cur_time is None:
                    st.caption("等待 Block D 帧选择…")
                    continue

                _any_shown = False
                for _entry in _qa_list:
                    _q  = _entry.get("question", {})
                    _q_matches = float(_q.get("time", -1)) == _cur_time
                    _r_matches = [
                        r for r in _entry.get("response", [])
                        if float(r.get("time", -1)) == _cur_time
                    ]
                    if not _q_matches and not _r_matches:
                        continue
                    _any_shown = True
                    if _q_matches:
                        st.markdown(
                            f"**🙋 Q** *(t={_q.get('time')}s)*  \n{_q.get('content', '')}",
                        )
                    for _r in _r_matches:
                        st.markdown(
                            f"**💬 A** *(t={_r.get('time')}s)*  \n{_r.get('content', '')}",
                        )
                    st.divider()
                if not _any_shown:
                    st.caption(f"当前帧 {_cur_time}s 无匹配 Q&A")

    st.divider()

    # ══ Block D: 图片序列交互展示区 ════════════════════════════════════════════

    paths = st.session_state.frame_paths
    idx   = st.session_state.display_indices
    total = len(paths)

    if not paths:
        st.info("暂无图片序列，请先加载含有 JSON 的文件夹")
    else:
        shown = [i + 1 for i in idx]
        pct   = (max(idx) + 1) / total if idx else 0.0

        # 状态栏：← 按钮 | 帧信息+进度 | → 按钮
        sc1, sc2, sc3 = st.columns([1, 8, 1])
        with sc1:
            st.button("←", key="vis_left",  on_click=nav_left,  use_container_width=True)
        with sc2:
            st.caption(
                f"当前帧: {shown if shown else '—'}  /  共 {total} 帧  ·  进度 {pct * 100:.0f}%"
            )
            st.progress(pct)
        with sc3:
            st.button("→", key="vis_right", on_click=nav_right, use_container_width=True)

        if not idx:
            st.info("按 **→** 开始浏览序列")
        else:
            # 最多 3 张图，动态列宽
            COLORS = ["#E74C3C", "#2980B9", "#27AE60"]
            img_cols = st.columns(len(idx))
            for ci, fi in enumerate(idx):
                with img_cols[ci]:
                    p    = paths[fi]
                    name = os.path.basename(p)
                    st.caption(f"帧 **{fi + 1}** / {total}")
                    if os.path.isfile(p):
                        img_bgr = cv2.imread(p)
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=name, use_container_width=True)
                        else:
                            st.warning(f"无法读取图片: {name}")
                    else:
                        # 占位色块（真实图片不存在时的 Mock 展示）
                        clr = COLORS[ci % len(COLORS)]
                        st.markdown(
                            f"""
                            <div style="
                                background:{clr}18;
                                border:2px dashed {clr};
                                border-radius:10px;
                                height:200px;
                                display:flex;
                                flex-direction:column;
                                align-items:center;
                                justify-content:center;
                                color:{clr};
                                text-align:center;
                                padding:1rem;
                                font-family:monospace;
                            ">
                                <div style="font-size:3rem">🖼️</div>
                                <div style="font-weight:700;font-size:1rem">Frame {fi + 1}</div>
                                <div style="font-size:0.72rem;opacity:0.75;
                                            margin-top:6px;word-break:break-all">
                                    {name}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ════════════════════════════════════════════════════════════════════════════
# JS 注入：键盘方向键触发可见的 ← / → 按钮（key="vis_left" / "vis_right"）
# ════════════════════════════════════════════════════════════════════════════

components.html(
    """
    <script>
    (function() {
      "use strict";

      // 通过 data-testid 找到包含指定 key 的按钮并点击
      function clickVisBtn(btnKey) {
        const btns = window.parent.document.querySelectorAll("button");
        for (const b of btns) {
          // Streamlit 将 key 注入到按钮的 data-testid 属性中
          const wrap = b.closest('[data-testid="stButton"]');
          if (wrap && wrap.querySelector('[data-testid="baseButton-secondary"]') === b) {
            const txt = (b.innerText || b.textContent).trim();
            if ((btnKey === "left" && txt === "←") ||
                (btnKey === "right" && txt === "→")) {
              b.dispatchEvent(new MouseEvent("click", {bubbles: true, cancelable: true}));
              return true;
            }
          }
        }
        // 回退：按文字匹配
        for (const b of window.parent.document.querySelectorAll("button")) {
          const txt = (b.innerText || b.textContent).trim();
          if ((btnKey === "left" && txt === "←") ||
              (btnKey === "right" && txt === "→")) {
            b.dispatchEvent(new MouseEvent("click", {bubbles: true, cancelable: true}));
            return true;
          }
        }
        return false;
      }

      if (window._kbdHandler) {
        try { window.parent.document.removeEventListener("keydown", window._kbdHandler, true); } catch(e) {}
      }

      window._kbdHandler = function(e) {
        const tag = (e.target || {}).tagName || "";
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (e.key === "ArrowLeft")  { e.preventDefault(); clickVisBtn("left");  }
        if (e.key === "ArrowRight") { e.preventDefault(); clickVisBtn("right"); }
      };

      try {
        window.parent.document.addEventListener("keydown", window._kbdHandler, true);
      } catch(err) {
        document.addEventListener("keydown", window._kbdHandler, true);
      }
    })();
    </script>
    """,
    height=0,
)
