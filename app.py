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

import altair as alt
import cv2
import numpy as np
import pandas as pd
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
    "sel_mode":        False,       # 选择模式开关
    "selected":        [],          # 已勾选的 JSON 文件列表
    "frame_paths":     [],          # 当前 JSON 中的帧路径列表
    "display_indices": [],          # 当前窗口中显示的帧索引列表（最多 3 个）
    "json_data":       None,        # 当前加载的完整 JSON 数据
    "threshold":       0.5,         # 触发阈值 [0, 1]
    "_toast":          None,        # 待展示的 toast 消息
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

def response_matches_frame(r: dict, frame_time: float) -> bool:
    """判断 response 条目是否与给定帧时间匹配。

    优先逻辑：若 st_time 与 end_time 均非空，则判断 frame_time 是否落在
    [st_time, end_time] 区间内；否则退回到 time 字段精确匹配。
    """
    st_t  = r.get("st_time", "")
    end_t = r.get("end_time", "")
    if st_t != "" and end_t != "":
        try:
            return float(st_t) <= frame_time <= float(end_t)
        except (ValueError, TypeError):
            pass
    return float(r.get("time", -1)) == frame_time


def get_json_files(root: str) -> list:
    """返回 root 下所有直接 JSON 文件（排序后）。"""
    if not root or not os.path.isdir(root):
        return []
    return sorted(
        f for f in os.listdir(root)
        if f.lower().endswith(".json") and os.path.isfile(os.path.join(root, f))
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
    """当选中数目达到最大展示数时，从第一个选中的 JSON 文件加载帧路径。"""
    sel   = st.session_state.selected
    root  = st.session_state.root_dir
    max_d = st.session_state.max_display

    if len(sel) < max_d or not sel or not root:
        st.session_state.json_data       = None
        st.session_state.frame_paths     = []
        st.session_state.display_indices = []
        return

    jpath = os.path.join(root, sel[0])
    data  = load_json(jpath)
    st.session_state.json_data = data
    if data:
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

def on_threshold_change() -> None:
    st.session_state.threshold = st.session_state._threshold_widget


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

    # ══ Block B: 文件选择区 ═══════════════════════════════════════════════════
    st.subheader("📂 目录")

    # 选择模式开关
    n_sel   = len(st.session_state.selected)
    max_d   = st.session_state.max_display
    in_sel  = st.session_state.sel_mode
    btn_lbl = "🔴 退出选择模式" if in_sel else "🟢 开启选择模式"

    if st.button(btn_lbl, use_container_width=True, key="btn_sel_toggle"):
        st.session_state.sel_mode = not in_sel
        st.rerun()

    if in_sel:
        st.info(f"✅ 选择模式已开启 — 已选 **{n_sel}** / {max_d}", icon="☑️")

    # JSON 文件列表
    json_files = get_json_files(st.session_state.root_dir)

    if not json_files:
        if st.session_state.root_dir:
            st.warning("⚠️ 路径无效或无 JSON 文件")
        else:
            st.caption("请先填写数据路径")
    else:
        st.caption(f"共 {len(json_files)} 个 JSON 文件")
        for fname in json_files:
            is_sel = fname in st.session_state.selected

            fc1, fc2 = st.columns([0.08, 0.92])

            with fc1:
                if in_sel:
                    st.checkbox(
                        "",
                        value=is_sel,
                        key=f"chk_{fname}",
                        on_change=on_folder_check,
                        args=(fname,),
                        label_visibility="hidden",
                    )

            with fc2:
                icon  = "📄" if is_sel else "📃"
                style = "color:#27AE60;font-weight:600;" if is_sel else "color:#555;"
                st.markdown(
                    f'<p style="margin:2px 0;{style}">{icon} {fname}</p>',
                    unsafe_allow_html=True,
                )


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
        st.info("💡 请在左侧开启选择模式并勾选 JSON 文件")
    else:
        _c_cols = st.columns(len(_sel))
        for _ci, _fname in enumerate(_sel):
            with _c_cols[_ci]:
                st.markdown(
                    f'<div style="font-weight:700;font-size:1rem;'
                    f'border-bottom:2px solid #4A90D9;padding-bottom:4px;'
                    f'margin-bottom:8px">📄 {_fname}</div>',
                    unsafe_allow_html=True,
                )
                _jpath = os.path.join(_root, _fname)
                if not os.path.isfile(_jpath):
                    st.warning("文件不存在")
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
                        if response_matches_frame(r, _cur_time)
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


    # ══ Block E: 折线图展示区 ════════════════════════════════════════════════
    if paths:
        st.divider()

        # 图例配置：多选（max_display > 1）或自动选中（= 1）
        _sel_all = st.session_state.selected
        if len(_sel_all) > 1:
            _legend_files = st.multiselect(
                "📊 图例",
                options=_sel_all,
                default=_sel_all,
                key="_legend_widget",
            )
        else:
            _legend_files = _sel_all

        st.number_input(
            "📊 触发阈值（回车确认）",
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            value=st.session_state.threshold,
            key="_threshold_widget",
            on_change=on_threshold_change,
        )

        # 为每个图例文件、每帧计算得分：1 - </silence> logit，无匹配则 0.0
        _all_rows: list = []
        for _lf in _legend_files:
            _lpath = os.path.join(_root, _lf)
            _lqa   = get_qa_data(_lpath)
            for _i, _fp in enumerate(paths):
                _ft    = extract_frame_time(_fp)
                _score = 0.0
                if _ft is not None:
                    for _entry in _lqa:
                        for _r in _entry.get("response", []):
                            if response_matches_frame(_r, _ft):
                                _silence = _r.get("logits", {}).get("</silence>")
                                if _silence is not None:
                                    _score = 1.0 - float(_silence)
                                break
                _all_rows.append({"帧序号": _i + 1, "得分": _score, "文件": _lf})

        _thresh      = st.session_state.threshold
        _frame_ticks = list(range(1, len(paths) + 1))

        if _all_rows:
            _df_score   = pd.DataFrame(_all_rows)
            _chart_line = (
                alt.Chart(_df_score)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "帧序号:Q",
                        title="帧序号",
                        axis=alt.Axis(grid=False, values=_frame_ticks, tickMinStep=1),
                    ),
                    y=alt.Y(
                        "得分:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="触发分数",
                        axis=alt.Axis(grid=False),
                    ),
                    color=alt.Color("文件:N", legend=alt.Legend(title="图例")),
                    tooltip=["帧序号:Q", "得分:Q", "文件:N"],
                )
            )
        else:
            # 无图例选中时，显示空白占位折线
            _df_score   = pd.DataFrame({"帧序号": _frame_ticks, "得分": [0.0] * len(paths)})
            _chart_line = (
                alt.Chart(_df_score)
                .mark_line()
                .encode(
                    x=alt.X("帧序号:Q", title="帧序号",
                             axis=alt.Axis(grid=False, values=_frame_ticks, tickMinStep=1)),
                    y=alt.Y("得分:Q", scale=alt.Scale(domain=[0, 1]),
                             title="触发分数", axis=alt.Axis(grid=False)),
                )
            )

        _chart_thresh = (
            alt.Chart(pd.DataFrame({"阈值": [_thresh]}))
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y="阈值:Q")
        )

        # 当前帧竖线（红色实线，随 Block D 刷新）
        _cur_frame = idx[-1] + 1 if idx else None
        _layers = [_chart_line, _chart_thresh]
        if _cur_frame is not None:
            _layers.append(
                alt.Chart(pd.DataFrame({"当前帧": [_cur_frame]}))
                .mark_rule(color="red", strokeWidth=2)
                .encode(x="当前帧:Q")
            )

        st.altair_chart(
            alt.layer(*_layers).configure_view(strokeWidth=0),
            use_container_width=True,
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
