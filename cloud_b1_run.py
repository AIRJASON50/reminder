"""
云端 B1 动态策略选股脚本 (GitHub Actions 版)
独立于 cloud_daily_run.py，使用动态 J 阈值 + 均线多头 + 胜率回测 + 流通市值过滤。

用法:
    python cloud_b1_run.py           # 正常运行（交易日才执行）
    python cloud_b1_run.py --force   # 强制执行（跳过交易日判断）
"""
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# 确保项目根目录在 path 中（兼容本地和 CI）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.screener import calc_b1_signals, check_ma_bullish_alignment, check_b1_winrate

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（从环境变量读取，与 cloud_daily_run.py 共用同一组 secrets）
# ---------------------------------------------------------------------------
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_PAGE_ID = os.environ.get("NOTION_PAGE_ID", "")
SERVERCHAN_KEY = os.environ.get("SERVERCHAN_KEY", "")
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")


# ===========================================================================
# 1. 交易日判断
# ===========================================================================
def is_trade_day(date: datetime = None) -> bool:
    if date is None:
        date = datetime.now()
    try:
        import akshare as ak
        trade_dates = ak.tool_trade_date_hist_sina()
        trade_date_set = set(trade_dates["trade_date"].astype(str))
        return date.strftime("%Y-%m-%d") in trade_date_set
    except Exception as e:
        log.warning(f"获取交易日历失败: {e}, 改用工作日判断")
        return date.weekday() < 5


# ===========================================================================
# 2. 数据拉取 — BaoStock 多进程并发，包含创业板/科创板
# ===========================================================================
def _fetch_batch(args):
    """子进程：独立登录 baostock，拉取一批股票数据。"""
    import baostock as bs

    bs_codes, start_date, end_date, batch_id, min_bars = args

    lg = bs.login()
    print(f"[B1-Worker-{batch_id}] login: code={lg.error_code}, msg={lg.error_msg}", flush=True)
    print(f"[B1-Worker-{batch_id}] date range: {start_date} ~ {end_date}, stocks: {len(bs_codes)}", flush=True)
    if lg.error_code != "0":
        print(f"[B1-Worker-{batch_id}] login FAILED, returning empty", flush=True)
        return {}

    results = {}
    skipped = 0

    for i, bs_code in enumerate(bs_codes):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
        except Exception as e:
            if i < 3:
                print(f"[B1-Worker-{batch_id}] {bs_code} exception: {e}", flush=True)
            continue

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if i < 2:
            print(f"[B1-Worker-{batch_id}] {bs_code}: {len(rows)} rows", flush=True)

        if len(rows) < min_bars:
            skipped += 1
            continue

        df = pd.DataFrame(rows, columns=rs.fields)
        df.rename(columns={"date": "trade_date"}, inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        if len(df) < min_bars:
            skipped += 1
            continue

        pure_code = bs_code.split(".")[1]
        results[pure_code] = df.reset_index(drop=True)

    bs.logout()
    print(f"[B1-Worker-{batch_id}] done: {len(results)} loaded, {skipped} skipped", flush=True)
    return results


def fetch_all_stock_data_b1(days: int = 500) -> tuple[dict, dict]:
    """用 baostock 多进程并发拉取主板+创业板+科创板股票近 N 天日线数据。

    B1 策略需要 114(MIDJ) + 96(回测) + 7(持仓) ≈ 217 交易日，500 日历日足够。
    股票池扩展：00/60/300/688 开头，排除 ST。

    Returns:
        all_data: {code: DataFrame}
        names:    {code: name}
    """
    import baostock as bs
    from multiprocessing import Pool

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    log.info("获取股票列表（含创业板/科创板）...")
    rs_basic = bs.query_stock_basic()
    basic_rows = []
    while rs_basic.error_code == "0" and rs_basic.next():
        basic_rows.append(rs_basic.get_row_data())
    basic_df = pd.DataFrame(basic_rows, columns=rs_basic.fields)

    # 过滤：上市股票、主板+创业板+科创板、排除ST
    basic_df = basic_df[(basic_df["type"] == "1") & (basic_df["status"] == "1")]
    basic_df["pure_code"] = basic_df["code"].str.split(".").str[1]
    basic_df = basic_df[basic_df["pure_code"].str.match(r"^(00|60|300|688)\d+$")]
    basic_df = basic_df[~basic_df["code_name"].str.contains("ST", case=False, na=False)]

    names = dict(zip(basic_df["pure_code"], basic_df["code_name"]))
    bs_codes = basic_df["code"].tolist()
    bs.logout()
    log.info(f"待拉取股票: {len(bs_codes)} 只")

    # 分成 4 批，多进程并发拉取
    min_bars = 200
    n_workers = 4
    chunk_size = len(bs_codes) // n_workers + 1
    batches = []
    for i in range(n_workers):
        chunk = bs_codes[i * chunk_size: (i + 1) * chunk_size]
        if chunk:
            batches.append((chunk, start_date, end_date, i, min_bars))

    log.info(f"启动 {len(batches)} 个并发进程拉取数据...")
    all_data = {}
    with Pool(n_workers) as pool:
        for batch_result in pool.imap_unordered(_fetch_batch, batches):
            all_data.update(batch_result)
            log.info(f"  已完成一批，累计: {len(all_data)} 只")

    log.info(f"成功加载 {len(all_data)} 只股票数据")
    return all_data, names


# ===========================================================================
# 3. 流通市值获取
# ===========================================================================
def fetch_market_cap() -> dict:
    """用 akshare 获取全 A 股流通市值（单位：亿元）。

    失败时返回空字典，选股逻辑会跳过市值过滤而非整体失败。
    """
    try:
        import akshare as ak
        log.info("获取流通市值数据...")
        spot = ak.stock_zh_a_spot_em()
        cap_map = {}
        for _, row in spot.iterrows():
            code = str(row.get("代码", ""))
            cap = row.get("流通市值", None)
            if code and cap is not None and not pd.isna(cap):
                cap_map[code] = cap / 1e8  # 转为亿元
        log.info(f"获取流通市值: {len(cap_map)} 只")
        return cap_map
    except Exception as e:
        log.warning(f"获取流通市值失败: {e}，将跳过市值过滤")
        return {}


# ===========================================================================
# 4. 选股逻辑 — B1 动态策略
# ===========================================================================
def scan_b1_picks(all_data: dict, names: dict, cap_map: dict) -> list[dict]:
    """组合4个条件：BB1信号 + 多头排列 + 胜率>50%且波动<10 + 流通市值≥30亿"""
    results = []
    total = len(all_data)
    for idx, (code, df) in enumerate(all_data.items()):
        if (idx + 1) % 500 == 0:
            log.info(f"  B1扫描进度: {idx+1}/{total}")

        try:
            # 条件1: BB1 信号 — 最新一天 J 下穿动态阈值
            df = calc_b1_signals(df)
            if not df["BB1"].iloc[-1]:
                continue

            # 条件2: MA 多头排列 (MA21 > MA55 > MA144)
            if not check_ma_bullish_alignment(df):
                continue

            # 条件3: 历史胜率 > 50% 且波动 < 10%
            wr = check_b1_winrate(df, lookback=96, forward=7)
            if wr["count"] == 0 or wr["winrate"] <= 50 or wr["volatility"] >= 10:
                continue

            # 条件4: 流通市值 ≥ 30亿（cap_map 为空时跳过此过滤）
            if cap_map:
                cap = cap_map.get(code, 0)
                if cap < 30:
                    continue
            else:
                cap = 0

            last = df.iloc[-1]
            results.append({
                "code": code,
                "name": names.get(code, ""),
                "close": round(last["close"], 2),
                "J": round(last["J"], 1),
                "LP": round(last["LP"], 1) if not pd.isna(last["LP"]) else 0,
                "MIDJ": round(last["MIDJ"], 1) if not pd.isna(last["MIDJ"]) else 0,
                "胜率": f"{wr['winrate']}%",
                "波动": f"{wr['volatility']}%",
                "信号数": wr["count"],
                "流通市值": round(cap, 1),
                "trade_date": last["trade_date"],
            })
        except Exception as e:
            continue

    results.sort(key=lambda x: float(x["胜率"].replace("%", "")), reverse=True)
    return results


# ===========================================================================
# 5. 通知推送（复用同一 secrets）
# ===========================================================================
def format_picks_text(picks: list[dict], date_str: str) -> tuple[str, str]:
    title = f"B1策略 {date_str} - 命中 {len(picks)} 只"
    if not picks:
        return title, "今日无符合条件的股票。"

    lines = ["| 代码 | 名称 | 收盘 | J值 | LP | 胜率 | 波动 | 市值(亿) |",
             "|------|------|------|-----|----|------|------|----------|"]
    for p in picks:
        lines.append(
            f"| {p['code']} | {p['name']} | {p['close']} "
            f"| {p['J']} | {p['LP']} | {p['胜率']} | {p['波动']} | {p['流通市值']} |"
        )
    return title, "\n".join(lines)


def notify_serverchan(title: str, body: str):
    if not SERVERCHAN_KEY:
        log.warning("SERVERCHAN_KEY 未配置，跳过 Server酱推送")
        return
    resp = requests.post(
        f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send",
        data={"title": title[:32], "desp": body},
        timeout=15,
    )
    if resp.ok:
        log.info("Server酱推送成功")
    else:
        log.error(f"Server酱推送失败: {resp.status_code} {resp.text}")


def notify_pushplus(title: str, body: str):
    if not PUSHPLUS_TOKEN:
        log.warning("PUSHPLUS_TOKEN 未配置，跳过 PushPlus 推送")
        return
    resp = requests.post(
        "http://www.pushplus.plus/send",
        json={"token": PUSHPLUS_TOKEN, "title": title[:50],
              "content": body, "template": "markdown"},
        timeout=15,
    )
    if resp.ok:
        log.info("PushPlus推送成功")
    else:
        log.error(f"PushPlus推送失败: {resp.status_code} {resp.text}")


def update_notion(picks: list[dict], date_str: str):
    if not NOTION_TOKEN or not NOTION_PAGE_ID:
        log.warning("Notion 凭据未配置，跳过")
        return

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    blocks = []
    blocks.append({"type": "divider", "divider": {}})
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content":
            f"B1策略 {date_str} ({len(picks)}只)"}}]},
    })

    if not picks:
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "今日无中标股票"}}]},
        })
    else:
        t_rows = [{"type": "table_row", "table_row": {"cells": [
            [{"text": {"content": "代码"}}],
            [{"text": {"content": "名称"}}],
            [{"text": {"content": "收盘"}}],
            [{"text": {"content": "J"}}],
            [{"text": {"content": "LP"}}],
            [{"text": {"content": "胜率"}}],
            [{"text": {"content": "波动"}}],
            [{"text": {"content": "市值(亿)"}}],
        ]}}]
        for p in picks:
            t_rows.append({"type": "table_row", "table_row": {"cells": [
                [{"text": {"content": p["code"]}}],
                [{"text": {"content": p["name"]}}],
                [{"text": {"content": str(p["close"])}}],
                [{"text": {"content": str(p["J"])}}],
                [{"text": {"content": str(p["LP"])}}],
                [{"text": {"content": p["胜率"]}}],
                [{"text": {"content": p["波动"]}}],
                [{"text": {"content": str(p["流通市值"])}}],
            ]}})

        blocks.append({
            "type": "table",
            "table": {
                "table_width": 8,
                "has_column_header": True,
                "has_row_header": False,
                "children": t_rows,
            },
        })

    resp = requests.patch(
        f"https://api.notion.com/v1/blocks/{NOTION_PAGE_ID}/children",
        headers=headers,
        json={"children": blocks},
        timeout=30,
    )
    if resp.status_code == 200:
        log.info("Notion 更新成功")
    else:
        log.error(f"Notion 更新失败: {resp.status_code} {resp.text}")


# ===========================================================================
# 主流程
# ===========================================================================
def run():
    parser = argparse.ArgumentParser(description="云端 B1 动态策略选股")
    parser.add_argument("--force", action="store_true", help="跳过交易日判断")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("B1 动态策略任务启动")

    # 交易日判断
    if not args.force and not is_trade_day():
        log.info("今日非交易日，跳过")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 拉取数据
    log.info("[1/4] 在线拉取股票数据（含创业板/科创板）...")
    all_data, names = fetch_all_stock_data_b1()

    # 获取流通市值
    log.info("[2/4] 获取流通市值...")
    cap_map = fetch_market_cap()

    # 选股
    log.info("[3/4] 执行 B1 策略筛选...")
    picks = scan_b1_picks(all_data, names, cap_map)
    log.info(f"B1策略中标: {len(picks)} 只")
    for p in picks:
        log.info(f"  {p['code']} {p['name']} 收盘={p['close']} J={p['J']} LP={p['LP']} 胜率={p['胜率']}")

    # 推送
    log.info("[4/4] 推送通知...")
    title, body = format_picks_text(picks, date_str)
    notify_serverchan(title, body)
    notify_pushplus(title, body)
    update_notion(picks, date_str)

    log.info("B1 动态策略任务完成")


if __name__ == "__main__":
    run()
