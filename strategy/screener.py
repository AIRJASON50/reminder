"""
选股策略模块
"""
import pandas as pd
import numpy as np


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """计算 KDJ 指标

    Args:
        df: 必须包含 high, low, close 列，按日期升序排列
        n: RSV 周期 (默认9)
        m1: K 平滑周期 (默认3)
        m2: D 平滑周期 (默认3)

    Returns:
        添加了 K, D, J 列的 DataFrame
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()

    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    # 递推计算 K, D
    k = np.zeros(len(df))
    d = np.zeros(len(df))
    k[0] = 50
    d[0] = 50
    for i in range(1, len(df)):
        k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv.iloc[i]
        d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]

    df = df.copy()
    df["K"] = k
    df["D"] = d
    df["J"] = 3 * k - 2 * d
    return df


def check_yang_volume_double(df: pd.DataFrame, lookback: int = 30,
                              multiplier: float = 2.0) -> bool:
    """检查最近 lookback 个交易日内是否出现放量阳线

    放量定义: 收盘 > 开盘（收阳）且 成交量 >= 20日均量 * multiplier

    Args:
        df: 必须包含 open, close, volume 列，按日期升序排列
        lookback: 回看天数
        multiplier: 相对20日均量的倍数
    """
    # 取 lookback + 20 行来保证均量计算有足够数据
    window = df.tail(lookback + 20).copy()
    window["avg20"] = window["volume"].rolling(20, min_periods=10).mean()
    recent = window.tail(lookback)
    if len(recent) < 2:
        return False

    recent = recent.copy()
    recent["is_yang"] = recent["close"] > recent["open"]
    recent["is_surge"] = recent["volume"] >= recent["avg20"] * multiplier

    return bool((recent["is_yang"] & recent["is_surge"]).any())


def check_no_large_yin_volume(df: pd.DataFrame, lookback: int = 60,
                               multiplier: float = 2.0,
                               yin_ratio: float = 0.8) -> bool:
    """检查最近 lookback 天内，放量阳线之后没有大阴量

    条件: lookback天内所有阴线的成交量 < 放量阳线成交量 * yin_ratio
    放量阳线定义: 阳线且成交量 >= 20日均量 * multiplier
    如果有多根放量阳线，取最近一根的成交量作为基准。

    Returns:
        True 表示通过（没有大阴量），False 表示存在大阴量
    """
    window = df.tail(lookback + 20).copy()
    window["avg20"] = window["volume"].rolling(20, min_periods=10).mean()
    recent = window.tail(lookback)
    if len(recent) < 2:
        return False

    recent = recent.copy()
    recent["is_yang"] = recent["close"] > recent["open"]
    recent["is_surge"] = recent["volume"] >= recent["avg20"] * multiplier

    yang_surge_rows = recent[recent["is_yang"] & recent["is_surge"]]
    if yang_surge_rows.empty:
        return False

    # 取最近一根放量阳线的成交量作为基准
    yang_volume = yang_surge_rows["volume"].iloc[-1]
    threshold = yang_volume * yin_ratio

    # 检查 lookback 天内所有阴线
    yin_rows = recent[~recent["is_yang"]]
    if yin_rows.empty:
        return True

    return bool((yin_rows["volume"] < threshold).all())


def check_near_ma60(df: pd.DataFrame, days: int = 30, pct: float = 0.05) -> bool:
    """检查当前价格在 MA60 附近

    条件: 最新收盘价距 MA60 不超过 pct（上下均可）

    Args:
        df: 必须包含 close 列，按日期升序排列
        days: 未使用，保留接口兼容
        pct: 最大偏离比例（如 0.05 = 5%）

    Returns:
        True 表示通过
    """
    if len(df) < 60:
        return False

    df = df.copy()
    df["ma60"] = df["close"].rolling(60).mean()

    last_close = df["close"].iloc[-1]
    last_ma60 = df["ma60"].iloc[-1]
    deviation = abs(last_close - last_ma60) / last_ma60

    return deviation <= pct


def check_yang_ratio(df: pd.DataFrame, days: int = 60, ratio: float = 0.7) -> bool:
    """检查近 days 天阳线占比是否达到 ratio

    Args:
        df: 必须包含 open, close 列，按日期升序排列
        days: 回看天数
        ratio: 阳线最低占比（如 0.7 = 70%）

    Returns:
        True 表示通过
    """
    if len(df) < days:
        return False
    recent = df.tail(days)
    yang_count = (recent["close"] > recent["open"]).sum()
    return yang_count / len(recent) >= ratio


# ========== 新增辅助策略 ==========

def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
              signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标"""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_dif"] = df["ema_fast"] - df["ema_slow"]
    df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2
    return df


def check_macd_reversal(df: pd.DataFrame) -> bool:
    """MACD 柱状图从负变正，或连续 2 天缩小（动量反转信号）"""
    if len(df) < 30:
        return False
    df = calc_macd(df)
    hist = df["macd_hist"].iloc[-3:]
    if len(hist) < 3:
        return False
    # 柱状图从负转正
    if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
        return True
    # 柱状图为负但连续缩小（绝对值减小）
    if hist.iloc[-1] < 0:
        if abs(hist.iloc[-1]) < abs(hist.iloc[-2]) < abs(hist.iloc[-3]):
            return True
    return False


def calc_bollinger(df: pd.DataFrame, period: int = 20,
                   num_std: float = 2.0) -> pd.DataFrame:
    """计算布林带"""
    df = df.copy()
    df["boll_mid"] = df["close"].rolling(period).mean()
    df["boll_std"] = df["close"].rolling(period).std()
    df["boll_upper"] = df["boll_mid"] + num_std * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - num_std * df["boll_std"]
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    return df


def check_bollinger_squeeze(df: pd.DataFrame, pct_rank: float = 0.2,
                             near_lower_pct: float = 0.02) -> bool:
    """布林带收窄 + 价格在下轨附近

    Args:
        pct_rank: 带宽处于近60天最低的百分位（如0.2=最低20%）
        near_lower_pct: 收盘价距下轨的最大偏离比例
    """
    if len(df) < 80:
        return False
    df = calc_bollinger(df)
    recent60_width = df["boll_width"].tail(60)
    current_width = df["boll_width"].iloc[-1]
    if pd.isna(current_width):
        return False
    rank = (recent60_width < current_width).sum() / len(recent60_width)
    if rank > pct_rank:
        return False

    last_close = df["close"].iloc[-1]
    last_lower = df["boll_lower"].iloc[-1]
    if pd.isna(last_lower):
        return False
    deviation = abs(last_close - last_lower) / last_close
    return deviation <= near_lower_pct


def check_volume_shrink(df: pd.DataFrame, days: int = 5) -> bool:
    """检查最近 days 天成交量逐日递减（缩量回调）

    要求至少有 days-1 天满足递减（允许 1 天不减）
    """
    if len(df) < days:
        return False
    vols = df["volume"].tail(days).values
    decreasing = sum(1 for i in range(1, len(vols)) if vols[i] < vols[i-1])
    return decreasing >= days - 2  # 允许1天不减


def calc_chip_distribution(df: pd.DataFrame, lookback: int = 60,
                            decay: float = 0.95, bins: int = 100):
    """基于 OHLCV 估算筹码分布

    Args:
        lookback: 回看天数
        decay: 每日衰减系数（模拟老筹码换手）
        bins: 价格区间数

    Returns:
        (price_axis, chip_distribution, stats_dict)
    """
    recent = df.tail(lookback)
    if len(recent) < 10:
        return None, None, {}

    price_min = recent["low"].min()
    price_max = recent["high"].max()
    if price_max <= price_min:
        return None, None, {}

    price_axis = np.linspace(price_min, price_max, bins)
    chips = np.zeros(bins)
    step = (price_max - price_min) / bins

    for i in range(len(recent)):
        row = recent.iloc[i]
        age = len(recent) - 1 - i
        weight = (decay ** age) * row["volume"]

        low, high = row["low"], row["high"]
        if high <= low:
            idx = int((low - price_min) / step)
            idx = min(idx, bins - 1)
            chips[idx] += weight
        else:
            idx_lo = max(0, int((low - price_min) / step))
            idx_hi = min(bins - 1, int((high - price_min) / step))
            if idx_hi > idx_lo:
                per_bin = weight / (idx_hi - idx_lo + 1)
                chips[idx_lo:idx_hi+1] += per_bin

    total_chips = chips.sum()
    if total_chips == 0:
        return price_axis, chips, {}

    # 获利比例
    last_close = recent["close"].iloc[-1]
    close_idx = min(bins - 1, max(0, int((last_close - price_min) / step)))
    profit_ratio = chips[:close_idx+1].sum() / total_chips

    # 筹码集中度: 90% 筹码区间
    cumsum = np.cumsum(chips) / total_chips
    p5_idx = np.searchsorted(cumsum, 0.05)
    p95_idx = np.searchsorted(cumsum, 0.95)
    p5_price = price_axis[min(p5_idx, bins-1)]
    p95_price = price_axis[min(p95_idx, bins-1)]
    concentration = (p95_price - p5_price) / last_close if last_close > 0 else 1.0

    # 平均成本
    avg_cost = np.average(price_axis, weights=chips) if total_chips > 0 else last_close

    stats = {
        "profit_ratio": profit_ratio,
        "concentration_90": concentration,
        "avg_cost": avg_cost,
        "close": last_close,
    }
    return price_axis, chips, stats


def check_chip_concentrated(df: pd.DataFrame, max_concentration: float = 0.15,
                             min_profit_ratio: float = 0.7) -> bool:
    """检查筹码集中度和获利比例

    Args:
        max_concentration: 90%筹码区间宽度/股价 的最大值
        min_profit_ratio: 最低获利比例
    """
    _, _, stats = calc_chip_distribution(df)
    if not stats:
        return False
    return (stats["concentration_90"] <= max_concentration and
            stats["profit_ratio"] >= min_profit_ratio)


def screen_kdj_j_low_with_yang_volume(
    code_list: list[str],
    fetch_func,
    j_threshold: float = 10,
    lookback: int = 30,
    vol_multiplier: float = 2.0,
    min_bars: int = 60,
) -> list[dict]:
    """选股: KDJ J < threshold 且最近N天内有阳量倍量

    Args:
        code_list: 股票代码列表
        fetch_func: 获取日线数据的函数 fetch_func(code) -> DataFrame
        j_threshold: J 值阈值
        lookback: 阳量倍量回看天数
        vol_multiplier: 倍量倍数
        min_bars: 最少需要的K线数量（过滤新股）

    Returns:
        命中的股票列表 [{"code", "name", "J", "last_date", "yang_vol_date"}, ...]
    """
    results = []
    total = len(code_list)

    for i, code in enumerate(code_list):
        print(f"\r  扫描中 [{i+1}/{total}] {code}", end="", flush=True)
        try:
            df = fetch_func(code)
            if df is None or len(df) < min_bars:
                continue

            # 计算 KDJ
            df = calc_kdj(df)

            # 条件1: 最新一天 J < threshold
            last_j = df["J"].iloc[-1]
            if last_j >= j_threshold:
                continue

            # 条件2: 最近 lookback 天内有阳量倍量
            if not check_yang_volume_double(df, lookback, vol_multiplier):
                continue

            # 条件3: 最近 lookback 天内没有大阴量(>= 倍量阳线的0.8倍)
            if not check_no_large_yin_volume(df, lookback, vol_multiplier, 0.8):
                continue

            # 找到最近的放量阳线日期
            window = df.tail(lookback + 20).copy()
            window["avg20"] = window["volume"].rolling(20, min_periods=10).mean()
            recent = window.tail(lookback).copy()
            recent["is_yang"] = recent["close"] > recent["open"]
            recent["is_surge"] = recent["volume"] >= recent["avg20"] * vol_multiplier
            yang_vol_rows = recent[recent["is_yang"] & recent["is_surge"]]
            yang_vol_date = yang_vol_rows["trade_date"].iloc[-1] if not yang_vol_rows.empty else ""

            results.append({
                "code": code,
                "J": round(last_j, 2),
                "K": round(df["K"].iloc[-1], 2),
                "D": round(df["D"].iloc[-1], 2),
                "close": df["close"].iloc[-1],
                "last_date": df["trade_date"].iloc[-1],
                "yang_vol_date": yang_vol_date,
            })
        except Exception as e:
            continue

    print()  # 换行
    return results


# ========== B1 动态策略 ==========

def calc_b1_signals(df: pd.DataFrame) -> pd.DataFrame:
    """B1 动态信号计算

    流程: KDJ J 值 → MIDJ 三重均值 (28/57/114) → LP 动态阈值 → BB1 信号

    Args:
        df: 必须包含 high, low, close 列，按日期升序排列，至少 200 行

    Returns:
        添加了 J, MIDJ, LP, BB1 列的 DataFrame
    """
    df = calc_kdj(df)

    # MIDJ: J 值的三重均值 (28 → 57 → 114)
    df["MIDJ_28"] = df["J"].rolling(28, min_periods=14).mean()
    df["MIDJ_57"] = df["MIDJ_28"].rolling(57, min_periods=28).mean()
    df["MIDJ"] = df["MIDJ_57"].rolling(114, min_periods=57).mean()

    # LP: 动态阈值 = MIDJ - 1.5 * std(J, 28)
    j_std = df["J"].rolling(28, min_periods=14).std()
    df["LP"] = df["MIDJ"] - 1.5 * j_std

    # BB1: J 下穿 LP 时发出信号 (前一天 J >= LP, 当天 J < LP)
    df["BB1"] = (df["J"].shift(1) >= df["LP"].shift(1)) & (df["J"] < df["LP"])

    return df


def check_ma_bullish_alignment(df: pd.DataFrame) -> bool:
    """MA 多头排列判定: MA21 > MA55 > MA144

    Args:
        df: 必须包含 close 列，至少 144 行

    Returns:
        True 表示当前处于多头排列
    """
    if len(df) < 144:
        return False

    ma21 = df["close"].rolling(21).mean().iloc[-1]
    ma55 = df["close"].rolling(55).mean().iloc[-1]
    ma144 = df["close"].rolling(144).mean().iloc[-1]

    if pd.isna(ma21) or pd.isna(ma55) or pd.isna(ma144):
        return False

    return ma21 > ma55 > ma144


def check_b1_winrate(df: pd.DataFrame, lookback: int = 96,
                     forward: int = 7) -> dict:
    """历史胜率回测: 统计过去 lookback 天内 BB1 信号的盈利概率

    用 shift(-forward) 获取信号后 forward 天的收盘价，计算收益率。
    胜率 = 盈利次数 / 信号总数, 波动 = 收益率标准差。

    Args:
        df: 已经计算过 calc_b1_signals 的 DataFrame
        lookback: 回测窗口天数
        forward: 持仓天数

    Returns:
        {"winrate": float, "volatility": float, "count": int}
    """
    if "BB1" not in df.columns:
        df = calc_b1_signals(df)

    # 取回测窗口 (需要预留 forward 天给未来收益)
    if len(df) < lookback + forward:
        return {"winrate": 0.0, "volatility": 99.0, "count": 0}

    backtest = df.iloc[-(lookback + forward):-forward].copy()
    backtest["future_close"] = df["close"].shift(-forward).iloc[-(lookback + forward):-forward]
    backtest["future_ret"] = (backtest["future_close"] - backtest["close"]) / backtest["close"]

    signals = backtest[backtest["BB1"] == True]
    if len(signals) == 0:
        return {"winrate": 0.0, "volatility": 99.0, "count": 0}

    wins = (signals["future_ret"] > 0).sum()
    winrate = wins / len(signals) * 100
    volatility = signals["future_ret"].std() * 100

    return {
        "winrate": round(winrate, 1),
        "volatility": round(volatility, 1),
        "count": len(signals),
    }
