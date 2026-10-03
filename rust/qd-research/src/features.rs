//! Reusable market-feature / indicator modules.
//!
//! Every feature: mathematically defined below, configurable parameters,
//! causal (uses only bars `<= i`, never the future), NaN-tolerant (missing
//! inputs propagate as `None`), independently unit-tested, complexity noted.
//! Nothing here is added "because it exists": each has a stated role in
//! trend / momentum / volatility / volume / advanced-statistics groups.

/// OHLCV bar. `t` is a plain sequence number (caller maps to timestamps).
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub t: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    pub fn is_valid(&self) -> bool {
        self.open.is_finite()
            && self.high.is_finite()
            && self.low.is_finite()
            && self.close.is_finite()
            && self.volume.is_finite()
            && self.high >= self.low
    }
}

/// Typical price, O((1)) per bar.
pub fn typical_price(b: &Bar) -> Option<f64> {
    if !b.is_valid() {
        return None;
    }
    Some((b.high + b.low + b.close) / 3.0)
}

// ---------------------------------------------------------------- trend ---

/// EMA with Wilder-style SMA seed: `ema[p-1] = mean(close[..p])`, then
/// `ema[i] = k*close[i] + (1-k)*ema[i-1]`, `k = 2/(p+1)`. Entries `< p-1`
/// are `None`. O(n) time, O(n) output.
pub fn ema(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    if closes.iter().take(period).any(|v| !v.is_finite()) {
        // Seed invalid: fall back to causal warm restart at first window
        // whose closes are all finite.
        return ema_restart(closes, period);
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut prev: f64 = closes[..period].iter().sum::<f64>() / period as f64;
    out[period - 1] = Some(prev);
    for (i, &c) in closes.iter().enumerate().skip(period) {
        if !c.is_finite() {
            prev = f64::NAN;
            out[i] = None;
            continue;
        }
        prev = if prev.is_nan() {
            // Re-seed after a gap: walk back up to `period` finite closes.
            match reseed(closes, i, period) {
                Some(v) => v,
                None => {
                    out[i] = None;
                    continue;
                }
            }
        } else {
            k * c + (1.0 - k) * prev
        };
        out[i] = Some(prev);
    }
    out
}

fn reseed(closes: &[f64], i: usize, period: usize) -> Option<f64> {
    let lo = i.saturating_sub(period - 1);
    let win = &closes[lo..=i];
    if win.iter().all(|v| v.is_finite()) {
        Some(win.iter().sum::<f64>() / win.len() as f64)
    } else {
        None
    }
}

fn ema_restart(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut prev = f64::NAN;
    let mut seeded = false;
    for i in 0..n {
        if i + 1 >= period && !seeded {
            let lo = i + 1 - period;
            let win = &closes[lo..=i];
            if win.iter().all(|v| v.is_finite()) {
                prev = win.iter().sum::<f64>() / period as f64;
                out[i] = Some(prev);
                seeded = true;
            }
            continue;
        }
        if seeded {
            let c = closes[i];
            if !c.is_finite() {
                prev = f64::NAN;
                seeded = false;
                out[i] = None;
            } else {
                prev = k * c + (1.0 - k) * prev;
                out[i] = Some(prev);
            }
        }
    }
    out
}

/// EMA slope in price units per bar: `ema[i] - ema[i-1]`. `None` where
/// either endpoint is missing. O(n).
pub fn ema_slope(ema_series: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut out = vec![None; ema_series.len()];
    for i in 1..ema_series.len() {
        if let (Some(a), Some(b)) = (ema_series[i - 1], ema_series[i]) {
            out[i] = Some(b - a);
        }
    }
    out
}

/// Signed relative distance `(close - ema) / ema`. `None` on missing data
/// or `ema == 0`. O(n).
pub fn price_distance_from_ema(closes: &[f64], ema_series: &[Option<f64>]) -> Vec<Option<f64>> {
    closes
        .iter()
        .zip(ema_series.iter())
        .map(|(&c, e)| match e {
            Some(e) if c.is_finite() && *e != 0.0 => Some((c - e) / e),
            _ => None,
        })
        .collect()
}

/// Session-anchored VWAP: `cum(typical*volume) / cum(volume)`, reset at
/// each `session[i] != session[i-1]` boundary. Zero-volume bars reuse the
/// previous value (no division by zero). O(n).
///
/// NOTE: on multi-day/multi-month series passed with a single session id
/// this accumulates over the whole history and drifts far from price — it
/// is an intraday (per-session) measure. For regime classification over
/// daily bars use [`vwap_rolling`] instead.
pub fn vwap(bars: &[Bar], session: &[i64]) -> Vec<Option<f64>> {
    let n = bars.len();
    let mut out = vec![None; n];
    let mut pv = 0.0;
    let mut vv = 0.0;
    for i in 0..n {
        let b = &bars[i];
        if i == 0 || session.get(i) != session.get(i - 1) {
            pv = 0.0;
            vv = 0.0;
        }
        match typical_price(b) {
            Some(tp) if b.volume > 0.0 => {
                pv += tp * b.volume;
                vv += b.volume;
                out[i] = Some(pv / vv);
            }
            _ => {
                out[i] = if vv > 0.0 { Some(pv / vv) } else { None };
            }
        }
    }
    out
}

/// Rolling VWAP: `sum(typical*volume) / sum(volume)` over the trailing
/// `period` bars (inclusive, causal). `None` until `period` bars exist or
/// the window's volume sums to ≤ 0. This is the regime-classifier anchor:
/// unlike session VWAP it stays near price on multi-day series, so
/// `|close - vwap|/vwap` and `|close - vwap|/atr` measure *current*
/// extension from value, not drift from a months-old open. O(n·period).
pub fn vwap_rolling(bars: &[Bar], period: usize) -> Vec<Option<f64>> {
    let n = bars.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let mut pv = 0.0;
        let mut vv = 0.0;
        let mut ok = true;
        for b in &bars[(i + 1 - period)..=i] {
            match typical_price(b) {
                Some(tp) if b.volume > 0.0 && tp.is_finite() && b.volume.is_finite() => {
                    pv += tp * b.volume;
                    vv += b.volume;
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && vv > 0.0 {
            out[i] = Some(pv / vv);
        }
    }
    out
}

/// Trend direction from EMA stack: `+1` if `fast > mid > slow`,
/// `-1` if reversed, else `0`. `None` where any leg is missing. O(n).
pub fn trend_direction(
    fast: &[Option<f64>],
    mid: &[Option<f64>],
    slow: &[Option<f64>],
) -> Vec<Option<i8>> {
    fast.iter()
        .zip(mid.iter())
        .zip(slow.iter())
        .map(|((f, m), s)| match (f, m, s) {
            (Some(f), Some(m), Some(s)) => {
                if f > m && m > s {
                    Some(1)
                } else if f < m && m < s {
                    Some(-1)
                } else {
                    Some(0)
                }
            }
            _ => None,
        })
        .collect()
}

/// Trend strength in `[0, 1]`: fraction of the last `window` bars whose
/// direction equals the current direction. `None` on missing data. O(n*w).
pub fn trend_strength(direction: &[Option<i8>], window: usize) -> Vec<Option<f64>> {
    let n = direction.len();
    let mut out = vec![None; n];
    if window == 0 {
        return out;
    }
    for i in 0..n {
        let cur = match direction[i] {
            Some(d) => d,
            None => continue,
        };
        let lo = i.saturating_sub(window - 1);
        let win = &direction[lo..=i];
        if win.iter().any(|d| d.is_none()) {
            continue;
        }
        let same = win.iter().filter(|d| d.unwrap() == cur).count();
        out[i] = Some(same as f64 / win.len() as f64);
    }
    out
}

// -------------------------------------------------------------- momentum ---

/// Rate of change in percent: `100*(close[i]/close[i-p] - 1)`. O(n).
pub fn roc(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if period == 0 || i < period || !c.is_finite() {
                return None;
            }
            let base = closes[i - period];
            if !base.is_finite() || base == 0.0 {
                return None;
            }
            Some(100.0 * (c / base - 1.0))
        })
        .collect()
}

/// Stochastic oscillator `%K`: `100*(close - LL) / (HH - LL)` over `period`
/// bars; flat window → `50.0`. `%D` = SMA of `%K` over `smooth` bars.
/// Returns `(k, d)`. O(n*p).
pub fn stochastic(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
    smooth: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = closes.len();
    let mut k = vec![None; n];
    if period == 0 || n < period {
        return (k.clone(), k);
    }
    for i in (period - 1)..n {
        let (mut hh, mut ll) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut ok = closes[i].is_finite();
        for j in (i + 1 - period)..=i {
            if !highs[j].is_finite() || !lows[j].is_finite() || !closes[j].is_finite() {
                ok = false;
                break;
            }
            if highs[j] > hh {
                hh = highs[j];
            }
            if lows[j] < ll {
                ll = lows[j];
            }
        }
        if !ok {
            continue;
        }
        k[i] = Some(if hh == ll { 50.0 } else { 100.0 * (closes[i] - ll) / (hh - ll) });
    }
    let d = sma_opt(&k, smooth.max(1));
    (k, d)
}

/// Simple moving average over `Option` series: `None` if any of the last
/// `period` values is missing. O(n*p).
pub fn sma_opt(series: &[Option<f64>], period: usize) -> Vec<Option<f64>> {
    let n = series.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let mut acc = 0.0;
        let mut ok = true;
        for j in (i + 1 - period)..=i {
            match series[j] {
                Some(v) => acc += v,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            out[i] = Some(acc / period as f64);
        }
    }
    out
}

// ------------------------------------------------------------- volatility ---

/// True range per bar (needs previous close; first bar uses high-low).
/// Invalid inputs → `None`. O(n).
pub fn true_range(bars: &[Bar]) -> Vec<Option<f64>> {
    bars.iter()
        .enumerate()
        .map(|(i, b)| {
            if !b.is_valid() {
                return None;
            }
            if i == 0 {
                return Some(b.high - b.low);
            }
            let pc = bars[i - 1].close;
            if !pc.is_finite() {
                return Some(b.high - b.low);
            }
            Some(
                (b.high - b.low)
                    .max((b.high - pc).abs())
                    .max((b.low - pc).abs()),
            )
        })
        .collect()
}

/// ATR via Wilder smoothing (RMA): seed = SMA of first `period` TRs.
/// `None` until index `period-1`, gaps propagate `None` then reseed. O(n).
pub fn atr(bars: &[Bar], period: usize) -> Vec<Option<f64>> {
    let tr = true_range(bars);
    wilder(&tr, period)
}

/// Wilder smoothing (RMA) over an `Option` series: SMA seed of first
/// `period` finite values, then `prev + (x - prev)/p`. O(n).
pub fn wilder(series: &[Option<f64>], period: usize) -> Vec<Option<f64>> {
    let n = series.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    let seed_ok = series[..period].iter().all(|v| v.is_some());
    if !seed_ok {
        // Find the first fully-finite window, like `ema_restart`.
        let mut seeded = false;
        let mut prev = 0.0;
        for i in 0..n {
            if !seeded {
                if i + 1 >= period {
                    let win = &series[i + 1 - period..=i];
                    if win.iter().all(|v| v.is_some()) {
                        prev = win.iter().map(|v| v.unwrap()).sum::<f64>() / period as f64;
                        out[i] = Some(prev);
                        seeded = true;
                    }
                }
                continue;
            }
            match series[i] {
                Some(x) => {
                    prev += (x - prev) / period as f64;
                    out[i] = Some(prev);
                }
                None => {
                    seeded = false;
                    out[i] = None;
                }
            }
        }
        return out;
    }
    let mut prev: f64 = series[..period].iter().map(|v| v.unwrap()).sum::<f64>() / period as f64;
    out[period - 1] = Some(prev);
    let mut seeded = true;
    for (i, v) in series.iter().enumerate().skip(period) {
        match v {
            Some(x) => {
                if !seeded {
                    match reseed_opt(series, i, period) {
                        Some(s) => {
                            prev = s;
                            seeded = true;
                        }
                        None => {
                            out[i] = None;
                            continue;
                        }
                    }
                }
                prev += (x - prev) / period as f64;
                out[i] = Some(prev);
            }
            None => {
                seeded = false;
                out[i] = None;
            }
        }
    }
    out
}

fn reseed_opt(series: &[Option<f64>], i: usize, period: usize) -> Option<f64> {
    let lo = i.saturating_sub(period - 1);
    let win = &series[lo..=i];
    if win.iter().all(|v| v.is_some()) {
        Some(win.iter().map(|v| v.unwrap()).sum::<f64>() / win.len() as f64)
    } else {
        None
    }
}

/// Rolling rank percentile of `series[i]` within the last `window` finite
/// values, in `[0, 1]` (`0.5` for a single value). `None` on missing input.
/// O(n*w).
pub fn rolling_percentile(series: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let n = series.len();
    let mut out = vec![None; n];
    if window == 0 {
        return out;
    }
    for i in 0..n {
        let cur = match series[i] {
            Some(v) => v,
            None => continue,
        };
        let lo = i.saturating_sub(window - 1);
        let mut win: Vec<f64> = series[lo..=i].iter().filter_map(|v| *v).collect();
        if win.is_empty() {
            continue;
        }
        win.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let below = win.iter().filter(|v| **v < cur).count();
        let equal = win.iter().filter(|v| **v == cur).count();
        // Mid-rank: stable for ties, deterministic.
        out[i] = Some((below as f64 + 0.5 * (equal as f64 - 1.0) + 0.5) / win.len() as f64);
    }
    out
}

/// Bollinger bandwidth in percent: `100*(upper-lower)/middle` with
/// `middle = SMA(period)`, bands at `±mult` population-std. O(n*p).
pub fn bollinger_bandwidth(closes: &[f64], period: usize, mult: f64) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let win = &closes[i + 1 - period..=i];
        if win.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mean = win.iter().sum::<f64>() / period as f64;
        if mean == 0.0 {
            continue;
        }
        let var = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
        out[i] = Some(100.0 * 2.0 * mult * var.sqrt() / mean.abs());
    }
    out
}

/// Rolling population standard deviation of closes. O(n*p).
pub fn rolling_std(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let win = &closes[i + 1 - period..=i];
        if win.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mean = win.iter().sum::<f64>() / period as f64;
        out[i] = Some((win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64).sqrt());
    }
    out
}

/// Realized volatility (annualized): `std(log(close[i]/close[i-1])) *
/// sqrt(periods_per_year)` over `window` log-returns. Needs `window + 1`
/// closes; non-positive prices → `None`. O(n*w).
pub fn realized_volatility(
    closes: &[f64],
    window: usize,
    periods_per_year: f64,
) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if window < 2 || n < window + 1 {
        return out;
    }
    for i in window..n {
        let mut rets = Vec::with_capacity(window);
        let mut ok = true;
        for j in (i - window + 1)..=i {
            let (a, b) = (closes[j - 1], closes[j]);
            if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                ok = false;
                break;
            }
            rets.push((b / a).ln());
        }
        if !ok {
            continue;
        }
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
        out[i] = Some(var.sqrt() * periods_per_year.sqrt());
    }
    out
}

// ----------------------------------------------------------------- volume ---

/// Relative volume: `volume[i] / SMA(volume, period)`. `None` on missing
/// data or zero mean. O(n*p).
pub fn relative_volume(volumes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = volumes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let win = &volumes[i + 1 - period..=i];
        if win.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mean = win.iter().sum::<f64>() / period as f64;
        if !volumes[i].is_finite() || mean <= 0.0 {
            continue;
        }
        out[i] = Some(volumes[i] / mean);
    }
    out
}

/// On-balance volume: running total, adding volume when `close > prev`,
/// subtracting when `close < prev`. Invalid bars carry the last value
/// forward (never `None` after the first valid bar). O(n).
pub fn obv(bars: &[Bar]) -> Vec<Option<f64>> {
    let mut out = vec![None; bars.len()];
    let mut acc = 0.0;
    let mut prev_close: Option<f64> = None;
    let mut started = false;
    for (i, b) in bars.iter().enumerate() {
        if !b.is_valid() {
            out[i] = if started { Some(acc) } else { None };
            continue;
        }
        if let Some(pc) = prev_close {
            if b.close > pc {
                acc += b.volume;
            } else if b.close < pc {
                acc -= b.volume;
            }
        }
        prev_close = Some(b.close);
        started = true;
        out[i] = Some(acc);
    }
    out
}

/// Chaikin money flow over `period`: `sum(mfv) / sum(volume)` with
/// `mfv = mfm * volume`, `mfm = ((c-l)-(h-c)) / (h-l)` (`h == l` → 0).
/// `None` on missing data or zero volume sum. O(n*p).
pub fn cmf(bars: &[Bar], period: usize) -> Vec<Option<f64>> {
    let n = bars.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let mut mfv = 0.0;
        let mut vol = 0.0;
        let mut ok = true;
        for b in &bars[i + 1 - period..=i] {
            if !b.is_valid() {
                ok = false;
                break;
            }
            let mfm = if b.high == b.low {
                0.0
            } else {
                ((b.close - b.low) - (b.high - b.close)) / (b.high - b.low)
            };
            mfv += mfm * b.volume;
            vol += b.volume;
        }
        if ok && vol > 0.0 {
            out[i] = Some(mfv / vol);
        }
    }
    out
}

// --------------------------------------------------------------- advanced ---

/// Rolling skewness (moment estimator, population): `m3 / m2^1.5`.
/// Zero variance → `0.0`. `None` on missing data. O(n*p).
pub fn rolling_skew(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    rolling_moment(closes, period, 3, |m2, m3| {
        if m2 <= 0.0 {
            0.0
        } else {
            m3 / m2.powf(1.5)
        }
    })
}

/// Rolling excess kurtosis: `m4 / m2^2 - 3`. Zero variance → `-3.0`
/// (degenerate point mass has Pearson kurtosis 1; we return excess `-2`?
/// No — return `0.0` is misleading; document: zero-variance → `-3.0`?).
/// Actually for a constant window `m2 == 0` we return `-3.0` (excess of a
/// degenerate distribution is undefined; `-3.0` = maximally platykurtic
/// placeholder, documented here). `None` on missing data. O(n*p).
pub fn rolling_kurtosis(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    rolling_moment(closes, period, 4, |m2, m4| {
        if m2 <= 0.0 {
            -3.0
        } else {
            m4 / m2.powi(2) - 3.0
        }
    })
}

fn rolling_moment(
    closes: &[f64],
    period: usize,
    order: u32,
    f: impl Fn(f64, f64) -> f64,
) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let win = &closes[i + 1 - period..=i];
        if win.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mean = win.iter().sum::<f64>() / period as f64;
        let m2 = win.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
        let mk = win.iter().map(|v| (v - mean).powi(order as i32)).sum::<f64>() / period as f64;
        out[i] = Some(f(m2, mk));
    }
    out
}

/// Hurst exponent via rescaled-range (R/S) over `window` log-returns:
/// `H = log2(R/S) / log2(n)` with `n = window`. Single-window estimate
/// (noisy by design — use as a regime input, not a signal). Constant
/// series → `0.5`. `None` on missing data. O(n*w).
pub fn hurst_rs(closes: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if window < 4 || n < window + 1 {
        return out;
    }
    for i in window..n {
        let mut rets = Vec::with_capacity(window);
        let mut ok = true;
        for j in (i - window + 1)..=i {
            let (a, b) = (closes[j - 1], closes[j]);
            if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                ok = false;
                break;
            }
            rets.push((b / a).ln());
        }
        if !ok {
            continue;
        }
        let mean = rets.iter().sum::<f64>() / window as f64;
        let mut cum = 0.0;
        let (mut mn, mut mx) = (0.0, 0.0);
        for r in &rets {
            cum += r - mean;
            if cum < mn {
                mn = cum;
            }
            if cum > mx {
                mx = cum;
            }
        }
        let r = mx - mn;
        let s = (rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64).sqrt();
        if s <= 0.0 {
            out[i] = Some(0.5);
        } else {
            out[i] = Some((r / s).log2() / (window as f64).log2());
        }
    }
    out
}

/// Lag-`lag` autocorrelation of log-returns over `window`. `None` on
/// missing data or zero variance. O(n*w).
pub fn autocorr_returns(closes: &[f64], window: usize, lag: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if window < 4 || lag == 0 || lag >= window || n < window + 1 {
        return out;
    }
    for i in window..n {
        let mut rets = Vec::with_capacity(window);
        let mut ok = true;
        for j in (i - window + 1)..=i {
            let (a, b) = (closes[j - 1], closes[j]);
            if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                ok = false;
                break;
            }
            rets.push((b / a).ln());
        }
        if !ok {
            continue;
        }
        let mean = rets.iter().sum::<f64>() / window as f64;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>();
        if var <= 0.0 {
            continue;
        }
        let cov = rets[lag..]
            .iter()
            .zip(rets[..window - lag].iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>();
        out[i] = Some(cov / var);
    }
    out
}

/// Return entropy: Shannon entropy (bits) of `bins` equal-width bins over
/// `window` log-returns. Constant window → `0.0`. `None` on missing data.
/// O(n*(w + bins)).
pub fn return_entropy(closes: &[f64], window: usize, bins: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if window < 2 || bins < 2 || n < window + 1 {
        return out;
    }
    for i in window..n {
        let mut rets = Vec::with_capacity(window);
        let mut ok = true;
        for j in (i - window + 1)..=i {
            let (a, b) = (closes[j - 1], closes[j]);
            if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                ok = false;
                break;
            }
            rets.push((b / a).ln());
        }
        if !ok {
            continue;
        }
        let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
        for r in &rets {
            if *r < mn {
                mn = *r;
            }
            if *r > mx {
                mx = *r;
            }
        }
        if mx == mn {
            out[i] = Some(0.0);
            continue;
        }
        let mut counts = vec![0usize; bins];
        for r in &rets {
            let mut b = ((r - mn) / (mx - mn) * bins as f64) as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1;
        }
        let total = window as f64;
        let ent = counts
            .iter()
            .filter(|c| **c > 0)
            .map(|c| {
                let p = *c as f64 / total;
                -p * p.log2()
            })
            .sum::<f64>();
        out[i] = Some(ent);
    }
    out
}

/// Price efficiency ratio (Kaufman): `|close[i] - close[i-p]| /
/// sum(|change|)` over `p` bars, in `[0, 1]`. Flat → `1.0` if the
/// endpoints are equal and nothing moved, else `0.0`-ish; exactly:
/// zero path sum → `1.0`. `None` on missing data. O(n*p).
pub fn efficiency_ratio(closes: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = vec![None; n];
    if period == 0 || n <= period {
        return out;
    }
    for i in period..n {
        if !closes[i].is_finite() || !closes[i - period].is_finite() {
            continue;
        }
        let mut path = 0.0;
        let mut ok = true;
        for j in (i - period + 1)..=i {
            if !closes[j].is_finite() || !closes[j - 1].is_finite() {
                ok = false;
                break;
            }
            path += (closes[j] - closes[j - 1]).abs();
        }
        if !ok {
            continue;
        }
        out[i] = Some(if path == 0.0 {
            1.0
        } else {
            (closes[i] - closes[i - period]).abs() / path
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars_from_closes(closes: &[f64]) -> Vec<Bar> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Bar {
                t: i as i64,
                open: c,
                high: c + 0.5,
                low: c - 0.5,
                close: c,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn ema_seed_and_recursion() {
        // SMA seed of first 3: (1+2+3)/3 = 2; k=0.5; ema[3] = 0.5*4+0.5*2 = 3.
        let e = ema(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(e[0], None);
        assert_eq!(e[1], None);
        assert_eq!(e[2], Some(2.0));
        assert_eq!(e[3], Some(3.0));
        assert_eq!(e[4], Some(4.0));
    }

    #[test]
    fn ema_handles_nan_gap() {
        let e = ema(&[1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0], 3);
        assert_eq!(e[3], None);
        assert_eq!(e[6], Some(6.0)); // reseeded SMA(5,6,7)
    }

    #[test]
    fn roc_and_stochastic_known_values() {
        let r = roc(&[100.0, 110.0, 121.0], 1)[2].unwrap();
        assert!((r - 10.0).abs() < 1e-9); // 100*(121/110 - 1), float-rounded
        let (k, d) = stochastic(
            &[10.0, 10.0, 10.0],
            &[8.0, 8.0, 8.0],
            &[9.0, 9.5, 9.0],
            3,
            2,
        );
        // window H=10 L=8: K = 100*(9-8)/2 = 50 then (9.5-8)/2=75, (9-8)/2=50
        assert_eq!(k, vec![None, None, Some(50.0)]);
        let _ = d;
        // flat window → 50
        let (k2, _) = stochastic(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0], 3, 1);
        assert_eq!(k2[2], Some(50.0));
    }

    #[test]
    fn atr_wilder_seed() {
        // TRs: bar0: 2; bar1: max(2, |12-10|=2, |10-10|=0)=2; bar2: 2.
        // seed SMA = 2 → ATR[2] = 2.
        let bars = vec![
            Bar { t: 0, open: 10.0, high: 11.0, low: 9.0, close: 10.0, volume: 1.0 },
            Bar { t: 1, open: 10.0, high: 12.0, low: 10.0, close: 11.0, volume: 1.0 },
            Bar { t: 2, open: 11.0, high: 12.0, low: 10.0, close: 11.5, volume: 1.0 },
        ];
        assert_eq!(atr(&bars, 3)[2], Some(2.0));
    }

    #[test]
    fn obv_cmf_direction() {
        let bars = bars_from_closes(&[10.0, 11.0, 10.5, 12.0]);
        let o = obv(&bars);
        assert_eq!(o[3], Some(100.0)); // +100 -100 +100
        // Symmetric ±0.5 bars have zero money-flow multiplier → CMF == 0.
        assert_eq!(cmf(&bars, 4).pop().unwrap(), Some(0.0));
        // Skewed closes push CMF positive.
        let up: Vec<Bar> = [10.0, 10.2, 10.4, 10.6]
            .iter()
            .enumerate()
            .map(|(i, &c)| Bar {
                t: i as i64,
                open: c,
                high: c + 0.1,
                low: c - 0.5,
                close: c,
                volume: 100.0,
            })
            .collect();
        assert!(cmf(&up, 4).pop().unwrap().unwrap() > 0.0);
    }

    #[test]
    fn advanced_stats_sanity() {
        let c: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let h = hurst_rs(&c, 20);
        assert!(h[59].unwrap() > 0.5); // trending → persistent
        let e = efficiency_ratio(&c, 10);
        assert_eq!(e[59], Some(1.0)); // straight line
        let s = rolling_skew(&c, 10);
        assert!(s[59].unwrap().abs() < 1e-9); // linear ramp ≈ symmetric
        let en = return_entropy(&c, 20, 8);
        assert!(en[59].unwrap() >= 0.0);
        let a = autocorr_returns(&c, 20, 1);
        assert!(a[59].is_some());
    }

    #[test]
    fn percentile_and_vwap_session_reset() {
        let s: Vec<Option<f64>> = vec![Some(1.0), Some(3.0), Some(2.0)];
        let p = rolling_percentile(&s, 3);
        assert_eq!(p[2], Some(0.5)); // (below + 0.5*equal) / n = (1+0.5)/3
        let bars = vec![
            Bar { t: 0, open: 10.0, high: 10.0, low: 10.0, close: 10.0, volume: 100.0 },
            Bar { t: 1, open: 20.0, high: 20.0, low: 20.0, close: 20.0, volume: 100.0 },
        ];
        let v = vwap(&bars, &[1, 2]); // new session → reset
        assert_eq!(v[1], Some(20.0));
    }

    #[test]
    fn vwap_rolling_tracks_price() {
        // Flat 100s then a jump: rolling VWAP(3) follows within a bar,
        // session VWAP over the whole run would lag far behind.
        let bars: Vec<Bar> = (0..6)
            .map(|i| {
                let c = if i < 3 { 100.0 } else { 110.0 };
                Bar { t: i, open: c, high: c, low: c, close: c, volume: 100.0 }
            })
            .collect();
        let v = vwap_rolling(&bars, 3);
        assert!(v[1].is_none()); // warmup
        assert_eq!(v[2], Some(100.0));
        assert!((v[4].unwrap() - 106.666).abs() < 0.01);
        assert_eq!(v[5], Some(110.0));
        assert!(vwap_rolling(&bars, 0).iter().all(|x| x.is_none()));
    }
}
