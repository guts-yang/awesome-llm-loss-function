"""Chinese dashboard for LLM loss geometry. Formulas stay in English LaTeX."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluate import logit_slice, probability_grid
from core.guide import GUIDE
from core.sandbox import FormulaError, derivative_latex, plugin_from_formula, plugin_skeleton
from core.schema import FAMILY_LABELS, all_losses, get_loss
from core.toy_dynamics import parameter_grid, simulate
from viz.plots import arc_figure, contour_figure, slice_figure, surface_figure, trajectory_figure

GOAL_ALL = "全部目标"
FAMILY_ALL = "全部"
PAGES = ["目录", "工作区", "更多"]
MORE_PAGES = ["公式实验室", "贡献"]
PROBABILITY_STAGES = ["三维势能", "二维下降", "一维切面"]
PARAMETER_STAGES = ["参数势能", "优化轨迹", "弧长投影"]

MECHANISMS = [
    (
        "势能有没有底",
        "梯度上升的 log p 随 p→0 趋向负无穷，logit 推力趋向 1。NPO 的 logistic、SatImp 的 p^β、JensUn 的 JS 散度都会把这个推力收掉。",
    ),
    (
        "概率质量流去哪里",
        "压低目标 token 之后，质量可能铺到均匀噪声、堆到一个高频词，或者流到替代答案和拒答。WHP 和 Unilogit 直接指定去向，梯度上升不指定。",
    ),
    (
        "梯度和谁缠在一起",
        "遗忘方向 v_f 与保留方向 v_r 的夹角 φ 变小以后，⟨∇L_f, ∇L_r⟩ 变大。GradDiff 把两股力相加，BLUR 的正交步把沿保留梯度的分量去掉。",
    ),
]

st.set_page_config(page_title="Awesome LLM Loss Function", page_icon="∑", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 4.75rem; padding-bottom: 2.4rem; max-width: 1280px;}
    div[data-testid="stSidebar"] {background: #E7F0F0;}
    div[data-testid="stButton"] button,
    div[data-testid="stPills"] button {
        white-space: normal;
        height: auto;
        min-height: 2.4rem;
    }
    div[data-testid="stButton"] button p,
    div[data-testid="stPills"] button p {
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 253, 248, 0.92);
        border-color: #d7e3e0;
        border-radius: 12px;
    }
    .identity-bar {
        background: #E7F0F0;
        border: 1px solid #c9ddd8;
        border-radius: 12px;
        padding: 0.85rem 1rem 0.7rem;
        margin: 0.2rem 0 0.9rem;
    }
    .identity-title {font-size: 1.35rem; font-weight: 650; color: #1C2430; line-height: 1.35;}
    .identity-title span {font-weight: 560; color: #0F6E6E; margin-left: 0.45rem;}
    .identity-meta {margin-top: 0.45rem;}
    .badge {
        display: inline-block;
        background: #F7F4EE;
        color: #1C2430;
        border: 1px solid #d5ddd8;
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        margin: 0 0.35rem 0.25rem 0;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .badge-warn {background: #F8E6DC; border-color: #E2B8A6;}
    .summary-line {
        margin: 0.4rem 0 0.55rem;
        color: #1C2430;
        line-height: 1.45;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .paper-row {margin: 0.15rem 0 0.45rem;}
    .paper-row a {color: #0F6E6E; text-decoration: none;}
    .paper-row a:hover {text-decoration: underline;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_papers():
    path = ROOT / "data" / "paper_catalog.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("papers", [])


def related_papers(plugin, papers, limit=6):
    needles = [item.lower() for item in plugin.keywords]
    hits = []
    for paper in papers:
        haystack = " ".join([paper.get("title", ""), " ".join(paper.get("tags", []))]).lower()
        if any(needle in haystack for needle in needles):
            hits.append(paper)
    hits.sort(key=lambda item: item.get("year") or 0, reverse=True)
    return hits[:limit]


def loss_label(plugin) -> str:
    return f"{plugin.family_zh()} · {plugin.name}"


def selected_plugin():
    label = st.session_state.get("loss_select")
    for item in all_losses():
        if loss_label(item) == label:
            return item
    return all_losses()[0]


def bound_label(plugin) -> str:
    return "有下界" if plugin.bounded_below else "无下界"


def step_label(plugin) -> str:
    return "正交投影" if plugin.step_mode == "project" else "普通梯度"


def badge_html(plugin) -> str:
    bound_class = "badge" if plugin.bounded_below else "badge badge-warn"
    return (
        f"<span class='{bound_class}'>{bound_label(plugin)}</span>"
        f"<span class='badge'>{step_label(plugin)}</span>"
    )


def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "目录"
    if "more_page" not in st.session_state:
        st.session_state.more_page = "公式实验室"
    if "goal_name" not in st.session_state:
        st.session_state.goal_name = GOAL_ALL
    if "catalog_family" not in st.session_state:
        st.session_state.catalog_family = FAMILY_ALL
    if "loss_select" not in st.session_state:
        st.session_state.loss_select = loss_label(all_losses()[0])


def _open_loss(label: str):
    st.session_state.loss_select = label
    st.session_state.page = "工作区"


def _on_goal_change():
    name = st.session_state.get("goal_name")
    if not name or name == GOAL_ALL:
        return
    goal = next(item for item in GUIDE if item["goal"] == name)
    st.session_state.loss_select = loss_label(get_loss(goal["losses"][0]))


def render_sidebar():
    st.sidebar.caption("Awesome LLM Loss Function")
    st.sidebar.header("当前损失")
    if st.session_state.page == "更多":
        st.sidebar.caption("公式实验室和贡献说明不使用损失超参数。")
        plugin = selected_plugin()
        return plugin, plugin.defaults()

    labels = [loss_label(item) for item in all_losses()]
    st.sidebar.selectbox("损失函数", labels, key="loss_select")
    plugin = selected_plugin()
    params = plugin.defaults()
    if st.session_state.page != "工作区":
        return plugin, params

    st.sidebar.subheader("超参数")
    if not plugin.params:
        st.sidebar.caption("这个损失没有可调系数。")
    for spec in plugin.params:
        params[spec.key] = st.sidebar.slider(
            spec.label,
            min_value=spec.minimum,
            max_value=spec.maximum,
            value=spec.default,
            step=spec.step,
            help=spec.help_text or None,
            key=f"{plugin.id}-{spec.key}",
        )
    return plugin, params


def render_nav():
    st.segmented_control(
        "页面",
        PAGES,
        key="page",
        required=True,
        label_visibility="collapsed",
        width="stretch",
    )
    if st.session_state.page == "更多":
        st.segmented_control(
            "更多",
            MORE_PAGES,
            key="more_page",
            required=True,
            label_visibility="collapsed",
            width="content",
        )


def render_identity(plugin):
    st.markdown(
        f"""
        <div class="identity-bar">
          <div class="identity-title">{html.escape(plugin.name_zh)}<span>{html.escape(plugin.name)}</span></div>
          <div class="identity-meta">
            <span class="badge">{html.escape(plugin.family_zh())}</span>
            <span class="badge">{plugin.year}</span>
            {badge_html(plugin)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_paper_list(papers):
    if not papers:
        st.caption("索引里还没有匹配的论文。可以先运行 python scripts/sync_catalog.py。")
        return
    for paper in papers:
        year = paper.get("year") or "年份待核"
        title = html.escape(paper.get("title", ""))
        url = html.escape(paper.get("url", ""), quote=True)
        code = ""
        if paper.get("code_url"):
            code_url = html.escape(paper["code_url"], quote=True)
            code = f" · <a href='{code_url}'>代码</a>"
        st.markdown(
            f"<div class='paper-row'><span class='badge'>{html.escape(str(year))}</span>"
            f"<a href='{url}'>{title}</a>{code}</div>",
            unsafe_allow_html=True,
        )


def render_goal_strip():
    st.pills(
        "按目标",
        [GOAL_ALL] + [item["goal"] for item in GUIDE],
        key="goal_name",
        required=True,
        on_change=_on_goal_change,
        label_visibility="collapsed",
        width="stretch",
        wrap=True,
    )


def _visible_losses(family: str):
    items = all_losses()
    if family != FAMILY_ALL:
        items = [item for item in items if item.family_zh() == family]
    goal_name = st.session_state.goal_name
    if goal_name != GOAL_ALL:
        goal = next(item for item in GUIDE if item["goal"] == goal_name)
        allowed = set(goal["losses"])
        items = [item for item in items if item.id in allowed]
    return items


def _render_loss_card(plugin):
    with st.container(border=True):
        st.markdown(f"**{plugin.name}** · {plugin.name_zh} · {plugin.year}")
        st.markdown(badge_html(plugin), unsafe_allow_html=True)
        st.markdown(f"<p class='summary-line'>{html.escape(plugin.summary_zh)}</p>", unsafe_allow_html=True)
        st.button(
            "打开",
            key=f"open-{plugin.id}",
            type="primary",
            on_click=_open_loss,
            args=(loss_label(plugin),),
            use_container_width=True,
        )


def show_catalog(plugin, papers):
    st.caption("按目标挑选，或按任务族浏览。打开卡片后进入这条损失的工作区。")
    render_goal_strip()
    goal_name = st.session_state.goal_name
    if goal_name != GOAL_ALL:
        goal = next(item for item in GUIDE if item["goal"] == goal_name)
        names = "、".join(get_loss(loss_id).name for loss_id in goal["losses"])
        st.write(f"先看 {names}。{goal['reason']}")
    family = st.radio(
        "任务族",
        [FAMILY_ALL] + [label for _, label in FAMILY_LABELS.items()],
        horizontal=True,
        key="catalog_family",
    )
    visible = _visible_losses(family)
    if not visible:
        st.info("当前目标和任务族下面没有损失。任务族选「全部」后，会列出这个目标里的损失。")
    grouped = {}
    for item in visible:
        grouped.setdefault(item.family_zh(), []).append(item)
    for family_name, items in grouped.items():
        st.subheader(f"{family_name} · {len(items)}")
        columns = st.columns(2)
        for index, item in enumerate(items):
            with columns[index % 2]:
                _render_loss_card(item)
    st.subheader("和当前损失相关的论文")
    show_paper_list(related_papers(plugin, papers))
    query = st.text_input("在论文索引里搜索", key="paper_query")
    if query:
        found = [paper for paper in papers if query.lower() in paper.get("title", "").lower()]
        st.caption(f"命中 {len(found)} 篇，显示最近的 20 篇。")
        show_paper_list(sorted(found, key=lambda paper: paper.get("year") or 0, reverse=True)[:20])


def compare_plugins(plugin):
    pool = [item for item in all_losses() if item.id != plugin.id and item.plane == plugin.plane]
    names = [item.name for item in pool]
    chosen = [name for name in st.session_state.get("compare_names", []) if name in names][:3]
    if st.session_state.get("compare_names") != chosen:
        st.session_state.compare_names = chosen
    selected = st.multiselect(
        "同屏对比（最多 3 个，使用各自默认超参数）",
        names,
        max_selections=3,
        key="compare_names",
    )
    return [item for item in pool if item.name in selected]


def _probability_plane(plugin):
    plane = "native"
    other = float(st.session_state.get("other_prob", 0.4))
    if plugin.supports_reference():
        choice = st.radio("坐标平面", ["任务平面", "参考模型平面"], horizontal=True, key="plane_choice")
        if choice == "参考模型平面":
            plane = "reference"
            other = st.slider("参考平面上的对照概率", 0.05, 0.95, 0.4, 0.01, key="other_prob")
    return plane, other


def _show_probability(plugin, params):
    plane, other = _probability_plane(plugin)
    stage = st.radio("阶段", PROBABILITY_STAGES, index=2, horizontal=True, key="geo_stage")
    y_slice = float(st.session_state.get("y_slice", 0.6))
    if stage != "三维势能":
        y_slice = st.slider("一维切面的纵轴位置", 0.05, 0.95, 0.6, 0.01, key="y_slice")
    x_label, y_label = plugin.axis_labels(plane)
    if stage == "一维切面":
        extras = compare_plugins(plugin)
        probability, loss, thrust = logit_slice(plugin, params, y_slice, mode=plane, other=other)
        curves = [{"name": plugin.name, "probability": probability, "loss": loss, "thrust": thrust}]
        for extra in extras:
            probability, loss, thrust = logit_slice(extra, extra.defaults(), y_slice, mode="native")
            curves.append({"name": extra.name, "probability": probability, "loss": loss, "thrust": thrust})
        st.plotly_chart(slice_figure(curves, x_label), width="stretch")
        st.caption("推力是 |∂L/∂z|，z = logit(p)。梯度下降使用带符号的 ∂L/∂z，悬停可以看到符号。GA 在低概率处推力停在高位；NPO、SimNPO、SatImp、JensUn 会衰减。")
        return
    x_grid, y_grid, values = probability_grid(plugin, params, n=51, mode=plane, other=other)
    if stage == "三维势能":
        st.plotly_chart(surface_figure(x_grid, y_grid, values, x_label, y_label, y_slice), width="stretch")
        st.caption("横轴和纵轴都是概率，高度是最小化的损失。")
        return
    st.plotly_chart(contour_figure(x_grid, y_grid, values, x_label, y_label, y_slice), width="stretch")
    st.caption("箭头是损失下降方向，虚线是当前一维切面。")


def _show_parameter(plugin, params):
    optimizer_label = st.selectbox(
        "优化器",
        ["SGD", "Adam", "正交投影"],
        index=2 if plugin.step_mode == "project" else 0,
        key=f"optimizer-{plugin.id}",
    )
    optimizer = {"SGD": "sgd", "Adam": "adam", "正交投影": "projected"}[optimizer_label]
    phi = st.slider("特征纠缠角 φ（度）", 5, 90, 30, 1, key="phi_deg")
    steps = st.slider("优化步数", 10, 80, 30, 5, key="opt_steps")
    learning_rate = st.slider("学习率", 0.02, 0.4, 0.12, 0.02, key="learning_rate")
    stage = st.radio("阶段", PARAMETER_STAGES, index=1, horizontal=True, key="param_stage")
    if stage == "参数势能":
        theta_1, theta_2, values = parameter_grid(plugin, phi, params, n=41)
        st.plotly_chart(surface_figure(theta_1, theta_2, values, "θ1", "θ2", title="参数势能"), width="stretch")
    else:
        extras = compare_plugins(plugin)
        runs = []
        for item, item_params in [(plugin, params)] + [(extra, extra.defaults()) for extra in extras]:
            result = simulate(
                item,
                item_params,
                phi_degrees=phi,
                optimizer=optimizer,
                steps=steps,
                learning_rate=learning_rate,
            )
            runs.append({"name": item.name, "result": result})
        chart = trajectory_figure(runs) if stage == "优化轨迹" else arc_figure(runs)
        st.plotly_chart(chart, width="stretch")
    st.caption("遗忘特征固定在 θ1 轴。保留特征与它夹角为 φ。φ 越小，两件事越缠在一起，普通梯度越容易把保留概率一起带走。正交投影会去掉沿保留交叉熵梯度的分量。")


def show_workspace(plugin, params, papers):
    reading, geometry = st.columns([4, 8])
    with reading:
        st.latex(plugin.latex)
        if plugin.toy_note_zh:
            st.caption(plugin.toy_note_zh)
        st.write(plugin.summary_zh)
        st.write(plugin.intuition_zh)
        with st.expander("统一的 logit 梯度"):
            st.write("把概率写成 p = σ(z) 之后，优化器真正加到 logit 上的力是损失对 z 的导数。概率平面上的陡坡还要再乘 p(1-p)，所以靠近 0 和 1 时，同样的概率导数不一定对应同样的 logit 推力。")
            st.latex(r"\frac{\partial \mathcal{L}}{\partial z}=\frac{\partial \mathcal{L}}{\partial p}\,p(1-p)= \mathcal{W}(p, p_{\mathrm{ref}}, \beta)\, d")
            st.write("权重 W 决定推力会不会在遗忘完成之后关掉。d 决定被推的是目标 token、对照 token，还是相对参考模型的对数比。")
        for title, body in MECHANISMS:
            with st.expander(title):
                st.write(body)
        st.subheader("当前损失的机制")
        st.write(plugin.physics_zh)
        st.subheader("它会怎么失败")
        st.write(plugin.failure_zh)
        with st.expander("相关论文"):
            show_paper_list(related_papers(plugin, papers, limit=6))
    with geometry:
        view = st.radio("视角", ["概率平面", "参数平面"], horizontal=True, key="lab_view")
        if view == "概率平面":
            _show_probability(plugin, params)
        else:
            _show_parameter(plugin, params)


def show_sandbox():
    st.title("公式实验室")
    st.write("输入 SymPy 能解析的表达式。可用符号是 x, y, beta, gamma, lam, eps。可用函数是 log, exp, sqrt, Abs, sigmoid, softplus。")
    example = "-(2/beta)*log(sigmoid(-beta*(log(x)-log(0.6))))"
    if "formula" not in st.session_state:
        st.session_state["formula"] = example
    formula = st.text_area("公式", key="formula", height=100)
    try:
        plugin = plugin_from_formula(formula)
        st.latex(plugin.latex)
        st.latex(r"\frac{\partial \mathcal{L}}{\partial x}=" + derivative_latex(formula))
        x_grid, y_grid, values = probability_grid(plugin, plugin.defaults(), n=41)
        probability, loss, thrust = logit_slice(plugin, plugin.defaults(), 0.6)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(surface_figure(x_grid, y_grid, values, "x", "y"), width="stretch")
        with right:
            st.plotly_chart(contour_figure(x_grid, y_grid, values, "x", "y", 0.6), width="stretch")
        st.plotly_chart(slice_figure([{"name": "自定义", "probability": probability, "loss": loss, "thrust": thrust}], "x"), width="stretch")
        st.code(plugin_skeleton(formula), language="python")
    except FormulaError as exc:
        st.error(str(exc))


def show_contribute():
    st.title("贡献")
    st.write("仓库是 awesome-llm-loss-function。新增一个损失时，在 core/losses/ 里注册插件，写中文机制说明，公式用英文 LaTeX。")
    st.latex(r"\mathcal{L}=\ldots")
    st.code(
        """@register_loss
class MyNovelLoss(LossPlugin):
    id = "my_novel_loss"
    name = "My Novel Margin-Orthogonal Loss"
    name_zh = "间隔正交损失"
    family = "unlearning"
    latex = r"\\mathcal{L} = ..."
    params = [ParamSpec("beta", "温度 beta", 0.2, 0.0, 2.0, 0.05)]

    def loss(self, x, y, params=None):
        x_arr, y_arr, merged = self.prepare(x, y, params)
        return x_arr + merged["beta"] * y_arr
""",
        language="python",
    )
    st.write("然后运行 pytest 和 python scripts/sync_catalog.py。论文索引来自旁边的 awesome-llm-unlearning，对齐与蒸馏的经典论文写在脚本的 curated 列表里。")
    st.markdown("远程仓库：[guts-yang/awesome-llm-loss-function](https://github.com/guts-yang/awesome-llm-loss-function)")
    st.markdown("条目组织参考了 [awesome-loss-functions](https://github.com/stabgan/awesome-loss-functions)，正文和代码没有从那里复制。")


def main():
    init_state()
    plugin, params = render_sidebar()
    render_nav()
    if st.session_state.page == "更多":
        if st.session_state.more_page == "公式实验室":
            show_sandbox()
        else:
            show_contribute()
        return
    render_identity(plugin)
    papers = load_papers()
    if st.session_state.page == "目录":
        show_catalog(plugin, papers)
    else:
        show_workspace(plugin, params, papers)


if __name__ == "__main__":
    main()
