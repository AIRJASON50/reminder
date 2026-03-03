"""
云端每日量化选股脚本 (GitHub Actions 版)
无需本地数据库，每次运行时在线拉取数据，完成筛选后推送通知。

用法:
    python cloud_daily_run.py           # 正常运行（交易日才执行）
    python cloud_daily_run.py --force   # 强制执行（跳过交易日判断）
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

from strategy.screener import calc_kdj

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
# 配置（从环境变量读取）
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
# 2. 数据拉取 — BaoStock 多进程并发拉取主板股票日线
# ===========================================================================
def _fetch_batch(args):
    """子进程：独立登录 baostock，拉取一批股票数据。"""
    import baostock as bs

    bs_codes, start_date, end_date, batch_id = args

    lg = bs.login()
    print(f"[Worker-{batch_id}] login: code={lg.error_code}, msg={lg.error_msg}", flush=True)
    if lg.error_code != "0":
        print(f"[Worker-{batch_id}] login FAILED, returning empty", flush=True)
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
                print(f"[Worker-{batch_id}] {bs_code} exception: {e}", flush=True)
            continue

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if i < 3:
            print(f"[Worker-{batch_id}] {bs_code}: {len(rows)} rows", flush=True)

        if len(rows) < 120:
            skipped += 1
            continue

        df = pd.DataFrame(rows, columns=rs.fields)
        df.rename(columns={"date": "trade_date"}, inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        if len(df) < 120:
            skipped += 1
            continue

        pure_code = bs_code.split(".")[1]
        results[pure_code] = df.reset_index(drop=True)

    bs.logout()
    print(f"[Worker-{batch_id}] done: {len(results)} loaded, {skipped} skipped", flush=True)
    return results


def fetch_all_stock_data(days: int = 250) -> tuple[dict, dict]:
    """用 baostock 多进程并发拉取所有主板股票近 N 天日线数据。

    需要 120 个交易日数据，250 日历日 ≈ 160 个交易日，留有余量。
    开 4 个进程并发拉取，约 15-20 分钟完成。

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

    # 获取股票列表和名称
    log.info("获取股票列表...")
    rs_basic = bs.query_stock_basic()
    basic_rows = []
    while rs_basic.error_code == "0" and rs_basic.next():
        basic_rows.append(rs_basic.get_row_data())
    basic_df = pd.DataFrame(basic_rows, columns=rs_basic.fields)

    # 过滤：上市股票、主板（00/60开头）、排除ST
    basic_df = basic_df[(basic_df["type"] == "1") & (basic_df["status"] == "1")]
    basic_df["pure_code"] = basic_df["code"].str.split(".").str[1]
    basic_df = basic_df[basic_df["pure_code"].str.match(r"^(00|60)\d{4}$")]
    basic_df = basic_df[~basic_df["code_name"].str.contains("ST", case=False, na=False)]

    names = dict(zip(basic_df["pure_code"], basic_df["code_name"]))
    bs_codes = basic_df["code"].tolist()
    bs.logout()
    log.info(f"待拉取股票: {len(bs_codes)} 只")

    # 分成 4 批，多进程并发拉取
    n_workers = 4
    chunk_size = len(bs_codes) // n_workers + 1
    batches = []
    for i in range(n_workers):
        chunk = bs_codes[i * chunk_size: (i + 1) * chunk_size]
        if chunk:
            batches.append((chunk, start_date, end_date, i))

    log.info(f"启动 {len(batches)} 个并发进程拉取数据...")
    all_data = {}
    with Pool(n_workers) as pool:
        for batch_result in pool.imap_unordered(_fetch_batch, batches):
            all_data.update(batch_result)
            log.info(f"  已完成一批，累计: {len(all_data)} 只")

    log.info(f"成功加载 {len(all_data)} 只股票数据")
    return all_data, names


# ===========================================================================
# 3. 选股逻辑（复用 daily_run.py 的 scan_picks）
# ===========================================================================
def scan_picks(all_data: dict, names: dict) -> list[dict]:
    """J<10 + 1.5x放量阳线 + 阴量<放量×0.7 + MA60上方0~5% + 无暴跌"""
    results = []
    for code, df in all_data.items():
        df = calc_kdj(df)
        df["avg20_vol"] = df["volume"].rolling(20, min_periods=10).mean()
        df["is_yang"] = df["close"] > df["open"]
        last = df.iloc[-1]

        # J < 10
        if last["J"] >= 10:
            continue

        # 资金进入: 30天内有1.5x放量阳线
        surge = df["is_yang"] & (df["volume"] >= df["avg20_vol"] * 1.5)
        if not surge.tail(30).any():
            continue
        surge_rows = df[surge].tail(30)
        last_surge_vol = surge_rows["volume"].iloc[-1]
        last_surge_date = surge_rows["trade_date"].iloc[-1]
        last_surge_ratio = last_surge_vol / df["avg20_vol"].iloc[-1]

        # 无出货: 60日最大阴量 < 放量×0.7
        yin_vol = df["volume"].where(~df["is_yang"], 0)
        max_yin_60d = yin_vol.tail(60).max()
        yin_ratio = max_yin_60d / last_surge_vol if last_surge_vol > 0 else 999
        if yin_ratio >= 0.7:
            continue

        # MA60上方 0~5%
        ma60 = df["close"].rolling(60).mean().iloc[-1]
        if pd.isna(ma60) or ma60 <= 0:
            continue
        ma60_dev = (last["close"] - ma60) / ma60
        if ma60_dev < 0 or ma60_dev > 0.05:
            continue

        # 无暴跌: 20天内无>=7%单日跌幅
        daily_ret = df["close"].pct_change()
        max_drop = daily_ret.tail(20).min()
        if max_drop <= -0.07:
            continue

        # RSI6
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss_s = (-delta).where(delta < 0, 0)
        ag6 = gain.rolling(6, min_periods=3).mean()
        al6 = loss_s.rolling(6, min_periods=3).mean()
        rsi6 = (100 - (100 / (1 + ag6 / al6.replace(0, np.nan)))).iloc[-1]

        # 连跌
        down = (df["close"] < df["close"].shift(1)).astype(int).values
        streak = 0
        for j in range(len(down) - 1, -1, -1):
            if down[j]:
                streak += 1
            else:
                break

        results.append({
            "code": code,
            "name": names.get(code, ""),
            "close": round(last["close"], 2),
            "J": round(last["J"], 1),
            "RSI6": round(rsi6, 1) if not pd.isna(rsi6) else 0,
            "MA60偏离": f"{ma60_dev:.1%}",
            "连跌": streak,
            "放量日": last_surge_date,
            "放量倍数": round(last_surge_ratio, 1),
            "阴阳比": round(yin_ratio, 2),
            "trade_date": last["trade_date"],
        })

    results.sort(key=lambda x: x["J"])
    return results


# ===========================================================================
# 4. 通知推送
# ===========================================================================
def format_picks_text(picks: list[dict], date_str: str) -> tuple[str, str]:
    """格式化选股结果为标题和正文"""
    title = f"量化选股 {date_str} - 命中 {len(picks)} 只"
    if not picks:
        return title, "今日无符合条件的股票。"

    lines = ["| 代码 | 名称 | 收盘 | J值 | MA60偏 | 放量 | 阴/阳 |",
             "|------|------|------|-----|--------|------|-------|"]
    for p in picks:
        lines.append(
            f"| {p['code']} | {p['name']} | {p['close']} "
            f"| {p['J']} | {p['MA60偏离']} | {p['放量倍数']}x | {p['阴阳比']} |"
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


# ===========================================================================
# 5. Notion 更新
# ===========================================================================
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
            f"{date_str} 选股结果 ({len(picks)}只)"}}]},
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
            [{"text": {"content": "MA60偏"}}],
            [{"text": {"content": "放量"}}],
            [{"text": {"content": "阴/阳"}}],
        ]}}]
        for p in picks:
            t_rows.append({"type": "table_row", "table_row": {"cells": [
                [{"text": {"content": p["code"]}}],
                [{"text": {"content": p["name"]}}],
                [{"text": {"content": str(p["close"])}}],
                [{"text": {"content": str(p["J"])}}],
                [{"text": {"content": p["MA60偏离"]}}],
                [{"text": {"content": f"{p['放量倍数']}x"}}],
                [{"text": {"content": str(p["阴阳比"])}}],
            ]}})

        blocks.append({
            "type": "table",
            "table": {
                "table_width": 7,
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
    parser = argparse.ArgumentParser(description="云端每日量化选股")
    parser.add_argument("--force", action="store_true", help="跳过交易日判断")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("云端量化任务启动")

    # 交易日判断
    if not args.force and not is_trade_day():
        log.info("今日非交易日，跳过")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 拉取数据
    log.info("[1/3] 在线拉取股票数据...")
    all_data, names = fetch_all_stock_data(days=150)

    # 选股
    log.info("[2/3] 执行选股筛选...")
    picks = scan_picks(all_data, names)
    log.info(f"今日中标: {len(picks)} 只")
    for p in picks:
        log.info(f"  {p['code']} {p['name']} 收盘={p['close']} J={p['J']}")

    # 推送
    log.info("[3/3] 推送通知...")
    title, body = format_picks_text(picks, date_str)
    notify_serverchan(title, body)
    notify_pushplus(title, body)
    update_notion(picks, date_str)

    log.info("云端量化任务完成")


if __name__ == "__main__":
    run()
