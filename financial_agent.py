"""
金融研报智能分析 Agent
======================
基于 Claude API 的自主式金融研究助手

核心能力：
  - 长链推理：将模糊研究问题分解为可执行的子任务序列
  - 财务分析：杜邦分解、财务比率计算、趋势分析、同业对比
  - 结构化输出：自动生成研究备忘录（Markdown 格式）

使用方法：
  export ANTHROPIC_API_KEY="sk-..."
  python financial_agent.py "分析贵州茅台近三年ROE的驱动因素"
  python financial_agent.py --batch  # 批量分析预设公司清单
"""

import os
import json
import sys
import argparse
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# ════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
DEFAULT_ANALYSIS_YEARS = 3


# ════════════════════════════════════════════════════════════════
#  数据模型
# ════════════════════════════════════════════════════════════════

@dataclass
class FinancialSnapshot:
    """单期财务数据快照"""
    year: int
    revenue: float            # 营业收入（亿元）
    net_profit: float         # 净利润（亿元）
    total_assets: float       # 总资产（亿元）
    total_equity: float       # 股东权益（亿元）
    operating_cost: float     # 营业成本（亿元）
    gross_profit: float       # 毛利（亿元）
    operating_profit: float   # 营业利润（亿元）
    cash_flow: float          # 经营活动现金流（亿元）
    current_assets: float     # 流动资产（亿元）
    current_liabilities: float # 流动负债（亿元）

@dataclass
class CompanyProfile:
    """公司资料"""
    name: str
    code: str                 # 股票代码
    industry: str             # 行业
    description: str = ""
    snapshots: list = field(default_factory=list)  # list[FinancialSnapshot]

    def latest(self) -> Optional[FinancialSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

@dataclass
class DupontResult:
    """杜邦分析结果"""
    year: int
    roe: float                # ROE = 净利润/股东权益
    net_profit_margin: float  # 净利率 = 净利润/营收
    asset_turnover: float     # 资产周转率 = 营收/总资产
    equity_multiplier: float  # 权益乘数 = 总资产/股东权益

@dataclass
class AnalysisReport:
    """完整分析报告"""
    company: str
    created_at: str = ""
    dupont_results: list = field(default_factory=list)
    ratios: dict = field(default_factory=dict)
    trends: dict = field(default_factory=dict)
    peer_comparison: dict = field(default_factory=dict)
    conclusion: str = ""


# ════════════════════════════════════════════════════════════════
#  示例公司财务数据（基于公开财报整理，已做简化处理）
# ════════════════════════════════════════════════════════════════

SAMPLE_DATA = [
    CompanyProfile(
        name="贵州茅台", code="600519.SH", industry="白酒",
        description="贵州茅台酒股份有限公司，主营茅台酒及系列酒生产销售",
        snapshots=[
            FinancialSnapshot(2022, 1275.5, 627.2, 2455.4, 1895.6, 119.3, 1156.2, 878.9, 368.2, 2185.5, 312.6),
            FinancialSnapshot(2023, 1505.6, 775.2, 2730.0, 2165.0, 133.9, 1371.7, 1033.7, 461.9, 2451.2, 334.5),
            FinancialSnapshot(2024, 1741.4, 918.5, 3050.2, 2480.3, 148.7, 1592.7, 1212.3, 520.1, 2760.4, 358.9),
        ]
    ),
    CompanyProfile(
        name="五粮液", code="000858.SZ", industry="白酒",
        description="宜宾五粮液股份有限公司，浓香型白酒龙头企业",
        snapshots=[
            FinancialSnapshot(2022, 739.7, 266.9, 1480.3, 1105.8, 161.2, 578.5, 397.6, 178.5, 1332.5, 238.7),
            FinancialSnapshot(2023, 832.7, 302.1, 1653.4, 1267.2, 179.8, 652.9, 456.3, 210.3, 1498.3, 265.8),
            FinancialSnapshot(2024, 931.2, 345.8, 1835.6, 1430.5, 198.5, 732.7, 518.6, 245.8, 1667.8, 295.3),
        ]
    ),
    CompanyProfile(
        name="美的集团", code="000333.SZ", industry="家电",
        description="美的集团股份有限公司，覆盖暖通空调、消费电器、机器人等业务",
        snapshots=[
            FinancialSnapshot(2022, 3457.1, 295.5, 4230.2, 1468.3, 2565.3, 891.8, 349.2, 316.8, 2852.6, 2157.8),
            FinancialSnapshot(2023, 3737.1, 337.2, 4520.5, 1635.5, 2764.8, 972.3, 394.5, 358.9, 3068.0, 2285.8),
            FinancialSnapshot(2024, 4075.3, 382.6, 4865.3, 1808.7, 3005.2, 1070.1, 438.2, 412.5, 3310.5, 2460.2),
        ]
    ),
]


# ════════════════════════════════════════════════════════════════
#  财务分析引擎
# ════════════════════════════════════════════════════════════════

class FinancialEngine:
    """财务分析工具集：不依赖外部 API，纯计算逻辑"""

    @staticmethod
    def dupont(snapshot: FinancialSnapshot) -> DupontResult:
        """杜邦分解：ROE = 净利率 × 资产周转率 × 权益乘数"""
        net_profit_margin = snapshot.net_profit / snapshot.revenue if snapshot.revenue else 0
        asset_turnover = snapshot.revenue / snapshot.total_assets if snapshot.total_assets else 0
        equity_multiplier = snapshot.total_assets / snapshot.total_equity if snapshot.total_equity else 0
        roe = snapshot.net_profit / snapshot.total_equity if snapshot.total_equity else 0
        return DupontResult(
            year=snapshot.year,
            roe=round(roe, 4),
            net_profit_margin=round(net_profit_margin, 4),
            asset_turnover=round(asset_turnover, 4),
            equity_multiplier=round(equity_multiplier, 4),
        )

    @staticmethod
    def ratios(snapshot: FinancialSnapshot) -> dict:
        """计算关键财务比率"""
        return {
            "毛利率": round((snapshot.gross_profit / snapshot.revenue * 100), 2) if snapshot.revenue else 0,
            "净利率": round((snapshot.net_profit / snapshot.revenue * 100), 2) if snapshot.revenue else 0,
            "ROE": round((snapshot.net_profit / snapshot.total_equity * 100), 2) if snapshot.total_equity else 0,
            "ROA": round((snapshot.net_profit / snapshot.total_assets * 100), 2) if snapshot.total_assets else 0,
            "资产负债率": round((1 - snapshot.total_equity / snapshot.total_assets) * 100, 2) if snapshot.total_assets else 0,
            "流动比率": round(snapshot.current_assets / snapshot.current_liabilities, 2) if snapshot.current_liabilities else 0,
            "营业利润率": round((snapshot.operating_profit / snapshot.revenue * 100), 2) if snapshot.revenue else 0,
            "经营现金流/营收": round((snapshot.cash_flow / snapshot.revenue * 100), 2) if snapshot.revenue else 0,
        }

    @staticmethod
    def trend_analysis(snapshots: list[FinancialSnapshot]) -> dict:
        """趋势分析：计算 CAGR 和逐年增长率"""
        if len(snapshots) < 2:
            return {"error": "数据不足，至少需要2期"}

        trends = {}
        years = [s.year for s in snapshots]
        n = len(snapshots) - 1

        metrics = {
            "营收": [s.revenue for s in snapshots],
            "净利润": [s.net_profit for s in snapshots],
            "总资产": [s.total_assets for s in snapshots],
            "净资产": [s.total_equity for s in snapshots],
        }

        for name, values in metrics.items():
            annual_changes = []
            for i in range(1, len(values)):
                if values[i-1]:
                    change = round((values[i] - values[i-1]) / values[i-1] * 100, 2)
                    annual_changes.append({"from": years[i-1], "to": years[i], "增幅": change})
            cagr = round((values[-1] / values[0]) ** (1/n) - 1, 4) * 100 if values[0] else 0
            trends[name] = {"CAGR": round(cagr, 2), "逐年变化": annual_changes}

        return trends

    @staticmethod
    def peer_comparison(companies: list[CompanyProfile], metric: str = "roe") -> list:
        """同业横向对比（指定指标最新一期）"""
        results = []
        for company in companies:
            latest = company.latest()
            if not latest:
                continue
            if metric == "roe":
                val = round(latest.net_profit / latest.total_equity * 100, 2) if latest.total_equity else 0
            elif metric == "net_margin":
                val = round(latest.net_profit / latest.revenue * 100, 2) if latest.revenue else 0
            elif metric == "revenue":
                val = round(latest.revenue, 2)
            else:
                val = 0
            results.append({"公司": company.name, metric: val, "年份": latest.year})
        return sorted(results, key=lambda x: x.get(metric, 0), reverse=True)

    @staticmethod
    def full_analysis(company: CompanyProfile) -> AnalysisReport:
        """对一家公司执行完整分析"""
        report = AnalysisReport(company=company.name)
        report.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 杜邦分析（逐年）
        for snap in company.snapshots:
            report.dupont_results.append(FinancialEngine.dupont(snap))

        # 最新一期财务比率
        if company.latest():
            report.ratios = FinancialEngine.ratios(company.latest())

        # 趋势分析
        report.trends = FinancialEngine.trend_analysis(company.snapshots)

        return report

    @staticmethod
    def format_report(report: AnalysisReport) -> str:
        """将分析格式化为 Markdown 文本"""
        lines = []
        lines.append(f"# {report.company} 财务分析报告")
        lines.append(f"生成时间：{report.created_at}\n")

        # 杜邦分析
        lines.append("## 一、杜邦分析")
        lines.append("| 年份 | ROE | 净利率 | 资产周转率 | 权益乘数 |")
        lines.append("|------|-----|--------|------------|----------|")
        for d in report.dupont_results:
            lines.append(f"| {d.year} | {d.roe*100:.2f}% | {d.net_profit_margin*100:.2f}% | {d.asset_turnover:.3f} | {d.equity_multiplier:.3f} |")

        if report.dupont_results and len(report.dupont_results) >= 2:
            first, last = report.dupont_results[0], report.dupont_results[-1]
            delta_roe = (last.roe - first.roe) * 100
            lines.append(f"\nROE 从 {first.year} 年的 {first.roe*100:.2f}% 变化至 {last.year} 年的 {last.roe*100:.2f}%，变动 {delta_roe:+.2f} 个百分点。")
            lines.append("")

        # 财务比率
        if report.ratios:
            lines.append("## 二、核心财务比率（最新一期）")
            lines.append("| 指标 | 数值 |")
            lines.append("|------|------|")
            for k, v in report.ratios.items():
                suffix = "%" if k not in ("流动比率", "资产周转率") else ""
                lines.append(f"| {k} | {v}{suffix} |")
            lines.append("")

        # 趋势分析
        if report.trends and "error" not in report.trends:
            lines.append("## 三、趋势分析")
            for metric, data in report.trends.items():
                lines.append(f"\n**{metric}**：CAGR = {data['CAGR']:.2f}%")
                for change in data.get("逐年变化", []):
                    arrow = "↑" if change["增幅"] > 0 else "↓"
                    lines.append(f"  - {change['from']}→{change['to']}：{change['增幅']:+.2f}% {arrow}")

        lines.append("")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  Agent 核心（结合 Claude API 进行长链推理）
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一名专业金融研究分析师。你的任务是对用户的研究问题进行深入分析。

请遵循以下流程：
1. 理解用户的研究问题
2. 将问题分解为具体的分析步骤（长链推理）
3. 对每个步骤给出专业见解
4. 结合数据形成最终结论

回复格式要求：
- 使用专业但清晰的语言
- 关键数据用 **加粗** 标注
- 适当使用表格和小标题
- 最终给出明确的投资研究结论

注意：你的输出将与实际的财务数据计算结果一起呈现给用户。"""


class FinancialAgent:
    """金融研报智能分析 Agent"""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        if Anthropic is None:
            sys.exit("缺少依赖：请运行 pip install anthropic")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if not self.client.api_key:
            sys.exit("错误：请设置 ANTHROPIC_API_KEY 环境变量或在初始化时传入 api_key")
        self.model = model
        self.engine = FinancialEngine()

    # ── 核心公开接口 ──

    def research(self, question: str, company_name: Optional[str] = None) -> str:
        """
        执行一项完整的金融研究任务。

        Args:
            question: 研究问题，例如 "分析贵州茅台近三年ROE的驱动因素"
            company_name: 目标公司名。不传则从已有数据中搜索

        Returns:
            Markdown 格式的完整研究报告
        """
        company = self._find_company(company_name or question)
        if not company:
            return self._no_data_response(question)

        # Phase 1: 执行定量分析（不依赖 API）
        report = self.engine.full_analysis(company)
        quantitative = self.engine.format_report(report)

        # Phase 2: 利用 Claude 做长链推理 + 定性分析
        reasoning = self._reasoning_chain(question, company, report)

        # Phase 3: 合成最终报告
        final_report = self._synthesize(question, company, quantitative, reasoning)
        return final_report

    def batch_research(self, companies: Optional[list[str]] = None) -> str:
        """批量分析多家公司并生成横向对比"""
        targets = []
        if companies:
            for name in companies:
                c = self._find_company(name)
                if c:
                    targets.append(c)
        else:
            targets = SAMPLE_DATA

        output = ["# 批量研究报告\n"]
        for c in targets:
            q = f"分析{c.name}的核心财务表现"
            output.append(self.research(q, c.name))
            output.append("\n---\n")

        # 横向对比
        if len(targets) >= 2:
            output.append("\n## 横向对比\n")
            for metric in ("roe", "net_margin", "revenue"):
                comp = self.engine.peer_comparison(targets, metric)
                metric_name = {"roe": "ROE(%)", "net_margin": "净利率(%)", "revenue": "营收(亿元)"}.get(metric, metric)
                output.append(f"\n### {metric_name}\n")
                output.append("| 公司 | 数值 | 年份 |")
                output.append("|------|------|------|")
                for item in comp:
                    output.append(f"| {item['公司']} | {item.get(metric, '')} | {item['年份']} |")

        return "\n".join(output)

    # ── 内部方法 ──

    def _find_company(self, query: str) -> Optional[CompanyProfile]:
        """从示例数据中匹配公司"""
        query_lower = query.lower()
        for c in SAMPLE_DATA:
            if c.name in query or c.code.lower() in query_lower:
                return c
        return None

    def _reasoning_chain(self, question: str, company: CompanyProfile, report: AnalysisReport) -> str:
        """利用 Claude 进行长链推理，解释财务数据背后的业务逻辑"""
        # 准备数据和问题
        dupont_table = ""
        for d in report.dupont_results:
            dupont_table += (
                f"  - {d.year}年：ROE={d.roe*100:.2f}%, "
                f"净利率={d.net_profit_margin*100:.2f}%, "
                f"资产周转率={d.asset_turnover:.3f}, 权益乘数={d.equity_multiplier:.3f}\n"
            )

        ratios_text = json.dumps(report.ratios, ensure_ascii=False, indent=2) if report.ratios else "无"

        trend_text = ""
        if report.trends and "error" not in report.trends:
            for metric, data in report.trends.items():
                trend_text += f"  {metric} CAGR: {data['CAGR']:.2f}%\n"

        user_prompt = f"""研究问题：{question}

公司背景：{company.name}（{company.code}）—— {company.description}

财务数据如下：

【杜邦分解】
{dupont_table}

【最新一期财务比率】
{ratios_text}

【趋势分析】
{trend_text}

请执行以下分析：
1. 拆解问题：该问题的核心要回答什么？
2. 因素归因：ROE 的三个驱动因素（净利率、资产周转率、权益乘数）中，哪个是主要驱动因素？
3. 业务解释：结合公司所处行业和业务特点，解释财务数据背后的业务逻辑
4. 风险提示：数据中反映出哪些潜在风险？
5. 研究结论：给出综合判断
"""
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text if msg.content else "（推理未产生结果）"
        except Exception as e:
            return f"（长链推理调用失败：{e}）"

    def _synthesize(self, question: str, company: CompanyProfile,
                    quantitative: str, reasoning: str) -> str:
        """合成定量分析 + 定性推理为最终报告"""
        return f"""# 金融研究备忘录

**研究问题**：{question}
**研究对象**：{company.name}（{company.code}）
**所属行业**：{company.industry}
**生成时间**：{datetime.now().strftime("%Y-%m-%d %H:%M")}

{"=" * 60}

## 第一部分：定量分析

{quantitative}

## 第二部分：定性分析（AI 推理）

{reasoning}

## 第三部分：综合结论

> 本报告基于 {company.name} 近 {len(company.snapshots)} 期财务数据，结合定量计算与定性分析生成。
> 数据来源为公司公开年报，分析结果仅供参考，不构成投资建议。
"""

    def _no_data_response(self, question: str) -> str:
        """数据缺失时的备选回复"""
        return f"""# 金融研究备忘录

**研究问题**：{question}

⚠️ 暂未找到该公司的财务数据。目前示例数据中包含：
{chr(10).join(f'  - {c.name}（{c.code}）' for c in SAMPLE_DATA)}

如需分析其他公司，可通过以下方式扩展：
1. 在 `SAMPLE_DATA` 中添加目标公司的财务数据
2. 使用 Web Search 工具动态获取数据（需联网功能支持）
"""


# ════════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="金融研报智能分析 Agent - 基于 Claude API 的研究助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  %(prog)s "分析贵州茅台近三年ROE的驱动因素"
  %(prog)s "分析五粮液的财务表现" --output report.md
  %(prog)s --batch
  %(prog)s --batch --companies "贵州茅台,美的集团"
        """,
    )
    parser.add_argument("question", nargs="?", help='研究问题，如"分析贵州茅台的ROE驱动因素"')
    parser.add_argument("--output", "-o", help="输出到文件（Markdown 格式）")
    parser.add_argument("--batch", action="store_true", help="批量分析模式")
    parser.add_argument("--companies", "-c", help="批量分析时指定公司，逗号分隔")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude 模型（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--list", action="store_true", help="列出示例数据中的公司")
    parser.add_argument("--local", action="store_true", help="本地模式（不调用 API，仅做数据分析计算）")

    args = parser.parse_args()

    # 列出公司（不需要 API Key）
    if args.list:
        print("示例数据中包含的公司：")
        for c in SAMPLE_DATA:
            snaps = len(c.snapshots)
            latest = c.latest()
            latest_info = f"，最新数据年份: {latest.year}" if latest else ""
            print(f"  {c.name}（{c.code}）—— {snaps}期数据{latest_info}")
        return

    # 本地模式（不需要 API Key，仅做数据分析计算）
    if args.local:
        if not args.question:
            parser.exit(1, "请提供研究问题。例如：--local \"分析贵州茅台\"\n")
        company_name = args.question.replace("分析", "").replace("的财务表现", "").replace("的ROE驱动因素", "").strip()
        company = None
        for c in SAMPLE_DATA:
            if c.name in company_name:
                company = c
                break
        if not company:
            available = "、".join(c.name for c in SAMPLE_DATA)
            parser.exit(1, f"未找到公司「{company_name}」，示例数据包含：{available}\n")
        engine = FinancialEngine()
        report = engine.full_analysis(company)
        result = engine.format_report(report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[OK] 报告已保存至: {args.output}")
        else:
            print(result)
        return

    # 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        parser.exit(1, "[ERROR] 请设置 ANTHROPIC_API_KEY 环境变量\n  export ANTHROPIC_API_KEY='sk-...'\n  （或无 API Key 时使用 --local 仅做数据分析）\n")

    agent = FinancialAgent(api_key=api_key, model=args.model)

    # 批量模式
    if args.batch:
        companies = args.companies.split(",") if args.companies else None
        result = agent.batch_research(companies)
    elif args.question:
        result = agent.research(args.question)
    else:
        parser.print_help()
        return

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[OK] 报告已保存至: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
