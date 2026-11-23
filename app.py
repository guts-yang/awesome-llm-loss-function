"""Chinese dashboard for LLM loss geometry. Formulas stay in English LaTeX."""

from __future__ import annotations

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

st.set_page_config(page_title="Awesome LLM Loss Function", page_icon="∑", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1180px;}
    div[data-testid="stSidebar"] {background: #E7F0F0;}
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


def render_formula(plugin):
    st.latex(plugin.latex)
    if plugin.toy_note_zh:
        st.caption(plugin.toy_note_zh)


def sidebar_controls():
    st.sidebar.header("看板")
    page = st.sidebar.radio("页面", ["损失目录", "地形实验室", "物理讲堂", "公式实验室", "贡献"], key="page")
    family = st.sidebar.selectbox("任务族", ["全部"] + [label for _, label in FAMILY_LABELS.items()])
    family_id = {label: key for key, label in FAMILY_LABELS.items()}.get(family)
    choices = [item for item in all_losses() if family_id is None or item.family == family_id]
    labels = {f"{item.family_zh()} · {item.name}": item.id for item in choices}
    selected_label = st.sidebar.selectbox("损失函数", list(labels))
    plugin = get_loss(labels[selected_label])
    params = plugin.defaults()
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
    plane = "native"
    if plugin.supports_reference():
        plane = "reference" if st.sidebar.radio("坐标平面", ["任务平面", "参考模型平面"]) == "参考模型平面" else "native"
    other = st.sidebar.slider("参考平面上的对照概率", 0.05, 0.95, 0.4, 0.01)
    y_slice = st.sidebar.slider("一维切面的纵轴位置", 0.05, 0.95, 0.6, 0.01)
    view = st.sidebar.radio("实验室视角", ["概率平面", "参数平面"])
    optimizer_label = st.sidebar.selectbox("优化器", ["SGD", "Adam", "正交投影"], index=2 if plugin.step_mode == "project" else 0)
    optimizer = {"SGD": "sgd", "Adam": "adam", "正交投影": "projected"}[optimizer_label]
    phi = st.sidebar.slider("特征纠缠角 φ（度）", 5, 90, 30, 1)
    steps = st.sidebar.slider("优化步数", 10, 80, 30, 5)
    learning_rate = st.sidebar.slider("学习率", 0.02, 0.4, 0.12, 0.02)
    compare_pool = [item for item in all_losses() if item.id != plugin.id and item.plane == plugin.plane]
    compare_labels = st.sidebar.multiselect(
        "同屏对比（最多 3 个，使用各自默认超参数）",
        [item.name for item in compare_pool],
        max_selections=3,
    )
    compare = [item for item in compare_pool if item.name in compare_labels]
    return {
        "page": page,
        "plugin": plugin,
        "params": params,
        "plane": plane,
        "other": other,
        "y_slice": y_slice,
        "view": view,
        "optimizer": optimizer,
        "phi": phi,
        "steps": steps,
        "learning_rate": learning_rate,
        "compare": compare,
    }


def show_catalog(plugin, papers):
    st.title("损失目录")
    st.write("这里只收 LLM 的损失：语言建模、偏好对齐、蒸馏、机器遗忘。每条都给出英文公式、中文机制说明，以及它在概率平面上的下界。")
    grouped = {}
    for item in all_losses():
        grouped.setdefault(item.family_zh(), []).append(item)
    for family_name, items in grouped.items():
        st.subheader(family_name)
        for item in items:
            with st.expander(f"{item.name} · {item.name_zh} · {item.year}"):
                st.latex(item.latex)
                st.write(item.summary_zh)
                bound = "有下界" if item.bounded_below else "无下界"
                st.caption(f"{bound}。更新方式：{'正交投影' if item.step_mode == 'project' else '普通梯度'}。")
                if item.paper_url:
                    st.markdown(f"论文：[{item.paper_title}]({item.paper_url})")
                if item.implementation_url:
                    st.markdown(f"实现：[{item.implementation_url}]({item.implementation_url})")
    st.subheader("按目标选择")
    for item in GUIDE:
        names = "、".join(get_loss(loss_id).name for loss_id in item["losses"])
        st.markdown(f"**{item['goal']}**")
        st.write(f"先看 {names}。{item['reason']}")
    st.subheader("和当前损失相关的论文")
    show_paper_list(related_papers(plugin, papers))
    query = st.text_input("在论文索引里搜索", key="paper_query")
    if query:
        found = [paper for paper in papers if query.lower() in paper.get("title", "").lower()]
        st.caption(f"命中 {len(found)} 篇，显示最近的 20 篇。")
        show_paper_list(sorted(found, key=lambda paper: paper.get("year") or 0, reverse=True)[:20])


def show_paper_list(papers):
    if not papers:
        st.caption("索引里还没有匹配的论文。可以先运行 python scripts/sync_catalog.py。")
        return
    for paper in papers:
        code = f" · 代码 {paper['code_url']}" if paper.get("code_url") else ""
        st.markdown(f"- [{paper['title']}]({paper['url']}) · {paper.get('year') or '年份待核'}{code}")


def show_lab(state):
    plugin = state["plugin"]
    st.title("地形实验室")
    st.write(f"{plugin.name_zh}（{plugin.name}）。横轴和纵轴都是概率，高度是最小化的损失。")
    render_formula(plugin)
    x_label, y_label = plugin.axis_labels(state["plane"])
    curves = []
    if state["view"] == "概率平面":
        x_grid, y_grid, values = probability_grid(plugin, state["params"], n=51, mode=state["plane"], other=state["other"])
        left, right = st.columns(2)
        with left:
            st.plotly_chart(surface_figure(x_grid, y_grid, values, x_label, y_label, state["y_slice"]), width="stretch")
        with right:
            st.plotly_chart(contour_figure(x_grid, y_grid, values, x_label, y_label, state["y_slice"]), width="stretch")
        probability, loss, thrust = logit_slice(plugin, state["params"], state["y_slice"], mode=state["plane"], other=state["other"])
        curves.append({"name": plugin.name, "probability": probability, "loss": loss, "thrust": thrust})
        for extra in state["compare"]:
            extra_params = extra.defaults()
            probability, loss, thrust = logit_slice(extra, extra_params, state["y_slice"], mode="native")
            curves.append({"name": extra.name, "probability": probability, "loss": loss, "thrust": thrust})
        st.plotly_chart(slice_figure(curves, x_label), width="stretch")
        st.caption("下图的推力是 |∂L/∂z|，z = logit(p)。梯度下降使用的是带符号的 ∂L/∂z，悬停可以看到符号。GA 在低概率处推力停在高位；NPO、SimNPO、SatImp、JensUn 会衰减。")
    else:
        theta_1, theta_2, values = parameter_grid(plugin, state["phi"], state["params"], n=41)
        runs = []
        for item, params in [(plugin, state["params"])] + [(extra, extra.defaults()) for extra in state["compare"]]:
            result = simulate(
                item,
                params,
                phi_degrees=state["phi"],
                optimizer=state["optimizer"],
                steps=state["steps"],
                learning_rate=state["learning_rate"],
            )
            runs.append({"name": item.name, "result": result})
        left, right = st.columns(2)
        with left:
            st.plotly_chart(surface_figure(theta_1, theta_2, values, "θ1", "θ2", title="参数平面上的损失"), width="stretch")
        with right:
            st.plotly_chart(trajectory_figure(runs), width="stretch")
        st.plotly_chart(arc_figure(runs), width="stretch")
        st.caption("遗忘特征固定在 θ1 轴。保留特征与它夹角为 φ。φ 越小，两件事越缠在一起，普通梯度越容易把保留概率一起带走。正交投影会去掉沿保留交叉熵梯度的分量。")


def show_lecture(plugin, papers):
    st.title("物理讲堂")
    st.write(f"当前例子是 {plugin.name_zh}。文字解释用中文，公式保持论文里的英文符号。")
    render_formula(plugin)
    columns = st.columns(3)
    columns[0].metric("任务族", plugin.family_zh())
    columns[1].metric("下界", "有" if plugin.bounded_below else "无")
    columns[2].metric("年份", plugin.year)
    st.subheader("这条损失在做什么")
    st.write(plugin.summary_zh)
    st.write(plugin.intuition_zh)
    st.subheader("统一的 logit 梯度")
    st.write("把概率写成 p = σ(z) 之后，优化器真正加到 logit 上的力是损失对 z 的导数。概率平面上的陡坡还要再乘 p(1-p)，所以靠近 0 和 1 时，同样的概率导数不一定对应同样的 logit 推力。")
    st.latex(r"\frac{\partial \mathcal{L}}{\partial z}=\frac{\partial \mathcal{L}}{\partial p}\,p(1-p)= \mathcal{W}(p, p_{\mathrm{ref}}, \beta)\, d")
    st.write("权重 W 决定推力会不会在遗忘完成之后关掉。d 决定被推的是目标 token、对照 token，还是相对参考模型的对数比。")
    st.subheader("三个机制")
    st.markdown("**势能有没有底。** 梯度上升的 log p 随 p→0 趋向负无穷，logit 推力趋向 1。NPO 的 logistic、SatImp 的 p^β、JensUn 的 JS 散度都会把这个推力收掉。")
    st.markdown("**概率质量流去哪里。** 压低目标 token 之后，质量可能铺到均匀噪声、堆到一个高频词，或者流到替代答案和拒答。WHP 和 Unilogit 直接指定去向，梯度上升不指定。")
    st.markdown("**梯度和谁缠在一起。** 遗忘方向 v_f 与保留方向 v_r 的夹角 φ 变小以后，⟨∇L_f, ∇L_r⟩ 变大。GradDiff 把两股力相加，BLUR 的正交步把沿保留梯度的分量去掉。")
    st.subheader("当前损失的机制")
    st.write(plugin.physics_zh)
    st.subheader("它会怎么失败")
    st.write(plugin.failure_zh)
    st.subheader("相关论文")
    show_paper_list(related_papers(plugin, papers, limit=8))


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
    state = sidebar_controls()
    papers = load_papers()
    page = state["page"]
    if page == "损失目录":
        show_catalog(state["plugin"], papers)
    elif page == "地形实验室":
        show_lab(state)
    elif page == "物理讲堂":
        show_lecture(state["plugin"], papers)
    elif page == "公式实验室":
        show_sandbox()
    else:
        show_contribute()


if __name__ == "__main__":
    main()
