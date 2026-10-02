//! Port of `backend_api_python/app/utils/numeric_precision.py`.
//!
//! Decimal helpers for exchange constraints and generated strategy source.
//!
//! Python works in `Decimal(str(value))`. This port mirrors CPython's
//! `Decimal` tuple semantics — `(sign, coefficient digits, exponent)` — with
//! integer arithmetic, so rounding *and* rendering match exactly:
//!
//! - division is correctly rounded to 28 significant digits, round-half-even
//!   (the CPython default context);
//! - `to_integral_value` keeps non-negative exponents untouched;
//! - rendering follows CPython's `Decimal.__str__` (plain vs `E` notation);
//! - the snap-tolerance test in `floor_decimal_to_step` is evaluated with
//!   exact integer comparisons, not floats.
//!
//! The single approximation: inputs whose integer arithmetic exceeds `u128`
//! fall back to an `f64` best-effort path. Real exchange steps, prices and
//! quantities are short strings and never hit it.

use std::cmp::{max, min};

/// Decimal value: `coeff * 10^exp`, `coeff >= 0`, sign in `neg`.
/// Mirrors CPython's `Decimal` `(sign, digits, exponent)` tuple.
#[derive(Debug, Clone, Copy)]
struct Dec {
    coeff: u128,
    exp: i32,
    neg: bool,
}

fn pow10(e: u32) -> Option<u128> {
    10u128.checked_pow(e)
}

/// Parse like Python `Decimal(str(value))`: plain or scientific notation.
/// Returns `None` for empty/garbage input (including `Infinity`/`NaN`,
/// which the Python callers also map to zero via their `except` branches).
fn parse(s: &str) -> Option<Dec> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (neg, rest) = match s.strip_prefix(['+', '-']) {
        Some(r) => (s.starts_with('-'), r),
        None => (false, s),
    };
    let (mantissa_part, exp_part) = match rest.find(['e', 'E']) {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest, None),
    };
    let mut exp: i32 = match exp_part {
        Some(e) => e.trim().parse().ok()?,
        None => 0,
    };
    let (int_part, frac_part) = match mantissa_part.find('.') {
        Some(i) => (&mantissa_part[..i], &mantissa_part[i + 1..]),
        None => (mantissa_part, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.bytes().all(|c| c.is_ascii_digit())
        || !frac_part.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let digits = digits.trim_start_matches('0');
    exp = exp.checked_sub(i32::try_from(frac_part.len()).ok()?)?;
    let coeff: u128 = if digits.is_empty() {
        0
    } else {
        digits.parse().ok()?
    };
    Some(Dec { coeff, exp, neg })
}

/// CPython `Decimal.__str__`: plain notation iff `exp <= 0` and at most six
/// digits would sit left of the point; otherwise scientific with `E±exp`.
fn dec_str(coeff: u128, exp: i32, neg: bool) -> String {
    let int_str = coeff.to_string();
    let leftdigits = exp as i64 + int_str.len() as i64;
    let dotplace: i64 = if exp <= 0 && leftdigits > -6 {
        leftdigits
    } else {
        1
    };
    let (intpart, fracpart) = if dotplace <= 0 {
        (
            "0".to_string(),
            format!(".{}{}", "0".repeat((-dotplace) as usize), int_str),
        )
    } else if dotplace as usize >= int_str.len() {
        (
            format!(
                "{}{}",
                int_str,
                "0".repeat(dotplace as usize - int_str.len())
            ),
            String::new(),
        )
    } else {
        let (a, b) = int_str.split_at(dotplace as usize);
        (a.to_string(), format!(".{b}"))
    };
    let expart = if leftdigits == dotplace {
        String::new()
    } else {
        format!("E{:+}", leftdigits - dotplace)
    };
    format!(
        "{}{}{}{}",
        if neg { "-" } else { "" },
        intpart,
        fracpart,
        expart
    )
}

/// `to_decimal` rendered the way Python's `str()` renders it.
/// Unparseable input becomes `"0"`.
pub fn to_decimal_str(value: &str) -> String {
    match parse(value) {
        Some(d) => dec_str(d.coeff, d.exp, d.neg),
        None => "0".to_string(),
    }
}

/// Exact `an * 10^ae / (bn * 10^be)` for positive operands, correctly rounded
/// to 28 significant digits with round-half-even (CPython default context).
/// Exact quotients keep every generated digit — CPython does not strip
/// trailing zeros from division results (`2.50 / 1` stays `2.50`).
fn dec_div(an: u128, ae: i32, bn: u128, be: i32) -> Option<(u128, i32)> {
    if bn == 0 {
        return None;
    }
    if an == 0 {
        return Some((0, ae.checked_sub(be)?));
    }
    let int_q = an / bn;
    let mut rem = an % bn;
    let mut digits = String::new();
    let mut frac_total: i64 = 0;
    let mut sig: u32 = 0;
    if int_q > 0 {
        digits.push_str(&int_q.to_string());
        sig = digits.len() as u32;
    }
    while sig < 29 && rem != 0 {
        rem = rem.checked_mul(10)?;
        let d = rem / bn;
        rem %= bn;
        frac_total += 1;
        if sig == 0 && d == 0 {
            continue; // leading fractional zero: shifts exponent, not significant
        }
        digits.push((b'0' + d as u8) as char);
        sig += 1;
    }
    let mut exp = (ae as i64 - be as i64) - frac_total;
    if digits.is_empty() {
        return Some((0, 0)); // unreachable for an > 0; defensive
    }
    if digits.len() > 28 {
        // Inexact quotient (always has >= 29 significant digits here), or an
        // over-precise exact integer: round to 28 significant digits,
        // half-even, tracking stickiness past the 29th digit.
        let sticky = rem != 0;
        let (keep_s, drop_s) = digits.split_at(28);
        let drop = drop_s.as_bytes();
        let rest_nonzero = drop[1..].iter().any(|&c| c != b'0') || sticky;
        let last_odd = keep_s.as_bytes()[27] % 2 == 1;
        let mut coeff: u128 = keep_s.parse().ok()?;
        if drop[0] > b'5' || (drop[0] == b'5' && (rest_nonzero || last_odd)) {
            coeff = coeff.checked_add(1)?;
        }
        exp += (digits.len() - 28) as i64;
        return Some((coeff, i32::try_from(exp).ok()?));
    }
    // Exact quotient (inexact ones always reach 29 significant digits above).
    let coeff: u128 = digits.parse().ok()?;
    Some((coeff, i32::try_from(exp).ok()?))
}

/// CPython `to_integral_value` for non-negative decimals: non-negative
/// exponents pass through untouched; otherwise round to an integer
/// (half-up or down) with exponent 0.
fn to_integral(coeff: u128, exp: i32, half_up: bool) -> Option<(u128, i32)> {
    if exp >= 0 {
        return Some((coeff, exp));
    }
    let shift = exp.unsigned_abs();
    if shift > 38 {
        // Division outputs carry <= 29 digits, so the value is far below 0.5
        // here; both modes round to zero.
        return Some((0, 0));
    }
    let divisor = pow10(shift)?;
    let qq = coeff / divisor;
    let rem = coeff % divisor;
    if rem == 0 {
        return Some((qq, 0));
    }
    if half_up {
        // divisor = 10^shift is even, so `2 * rem >= divisor` is the exact
        // half-up test; doubling fits (rem < divisor <= 1e38).
        let up = rem.checked_mul(2)? >= divisor;
        Some((qq.checked_add(if up { 1 } else { 0 })?, 0))
    } else {
        Some((qq, 0))
    }
}

/// Floor a positive value to a step, absorbing float-only edge noise.
///
/// Mirrors `floor_decimal_to_step`: values within one part per trillion of an
/// integer step are snapped to that integer before flooring; real below-step
/// quantities stay below the boundary. Returns the string Python's `str()`
/// would produce for the resulting `Decimal` (including `0.0000`-style zeros
/// and `1E-7`-style scientific notation).
pub fn floor_to_step_str(value: &str, step: &str) -> String {
    let number = match parse(value) {
        Some(d) => d,
        None => return "0".to_string(),
    };
    let increment = match parse(step) {
        Some(d) => d,
        None => return dec_str(number.coeff, number.exp, number.neg),
    };
    if number.coeff == 0 || number.neg {
        return "0".to_string();
    }
    if increment.coeff == 0 || increment.neg {
        return dec_str(number.coeff, number.exp, number.neg);
    }
    match floor_exact(number, increment) {
        Some(s) => s,
        None => f64_fallback_floor(number, increment),
    }
}

fn floor_exact(number: Dec, increment: Dec) -> Option<String> {
    let (uc, ue) = dec_div(
        number.coeff,
        number.exp,
        increment.coeff,
        increment.exp,
    )?;
    // Nearest integer, half-up, kept in Decimal tuple form.
    let (nc, ne) = to_integral(uc, ue, true)?;
    // Exact snap test: |units - nearest) <= max(1e-12, |units| * 1e-12).
    // With units = U / D in lowest integer terms this becomes
    // K * 10^12 <= D (first branch) or K * 10^12 <= U (second branch),
    // where K is the scaled absolute difference — all integer math.
    let snapped = if ue >= 0 {
        let u_val = uc.checked_mul(pow10(ue as u32)?)?;
        let m_val = nc.checked_mul(pow10(ne as u32)?)?;
        let k = u_val.abs_diff(m_val);
        k == 0 || matches!(k.checked_mul(1_000_000_000_000), Some(s) if s <= u_val)
    } else {
        let s = ue.unsigned_abs();
        let scaled_exp = match ne.checked_add_unsigned(s) {
            Some(v) => v,
            None => return None,
        };
        let m_scaled = match pow10(scaled_exp as u32)
            .and_then(|p| nc.checked_mul(p))
        {
            Some(v) => v,
            None => return None,
        };
        let k_num = uc.abs_diff(m_scaled);
        let cond1 = if s >= 12 {
            match pow10(s - 12) {
                Some(bound) => k_num <= bound,
                None => true, // bound exceeds u128: always satisfied
            }
        } else {
            k_num == 0
        };
        let cond2 = matches!(k_num.checked_mul(1_000_000_000_000), Some(x) if x <= uc);
        cond1 || cond2
    };
    let (wc, we) = if snapped {
        (nc, ne)
    } else {
        to_integral(uc, ue, false)?
    };
    let prod_coeff = wc.checked_mul(increment.coeff)?;
    let prod_exp = we.checked_add(increment.exp)?;
    Some(dec_str(prod_coeff, prod_exp, false))
}

/// Best-effort fallback for inputs beyond `u128` range (unreachable for real
/// exchange steps/prices). Mirrors the shape of the old float behavior.
fn f64_fallback_floor(number: Dec, increment: Dec) -> String {
    let dec_to_f64 = |d: Dec| -> f64 {
        dec_str(d.coeff, d.exp, d.neg).parse::<f64>().unwrap_or(0.0)
    };
    let (nf, sf) = (dec_to_f64(number), dec_to_f64(increment));
    if !(sf > 0.0) {
        return format!("{nf}");
    }
    let units = nf / sf;
    if !units.is_finite() || units <= 0.0 {
        return "0".to_string();
    }
    format!("{}", units.floor() * sf)
}

/// Quantize a decimal string with ROUND_HALF_UP to `places` (clamped 0..=18),
/// then convert to `f64` exactly like Python's `float(quantized Decimal)`.
/// Zero (including negative zero) maps to `0.0`; garbage maps to `0.0`.
pub fn clean_decimal_str(value: &str, decimal_places: i64) -> f64 {
    let places = min(18, max(0, decimal_places));
    let d = match parse(value) {
        Some(d) => d,
        // Beyond u128 range (or garbage): f64 multiply-and-round is exact
        // enough here — huge magnitudes have no fractional precision anyway.
        None => return f64_round(value, places),
    };
    if d.coeff == 0 {
        return 0.0;
    }
    // Target exponent is -places: round coeff * 10^(exp + places) to an
    // integer (half-up on the magnitude), then scale back.
    let shift = match d.exp.checked_add(places as i32) {
        Some(s) => s,
        None => return 0.0,
    };
    let q: u128 = if shift >= 0 {
        // Quantizing an integer-valued decimal to fractional places is the
        // identity — only the f64 conversion matters. Huge integers would
        // overflow u128 here, but the plain `str -> f64` parse is correctly
        // rounded, exactly like Python's `float(quantized Decimal)`.
        let p = match pow10(shift as u32) {
            Some(v) => v,
            None => return dec_str(d.coeff, d.exp, d.neg).parse::<f64>().unwrap_or(0.0),
        };
        match d.coeff.checked_mul(p) {
            Some(v) => v,
            None => return dec_str(d.coeff, d.exp, d.neg).parse::<f64>().unwrap_or(0.0),
        }
    } else {
        let neg_shift = shift.unsigned_abs();
        if neg_shift > 38 {
            // value far below half a unit at this precision: rounds to zero
            // (coeff has <= 39 digits, so coeff * 2 < 10^neg_shift).
            return 0.0;
        }
        let div = match pow10(neg_shift) {
            Some(v) => v,
            None => return f64_round(value, places),
        };
        let qq = d.coeff / div;
        let rem = d.coeff % div;
        match rem.checked_mul(2) {
            Some(double) if double >= div => match qq.checked_add(1) {
                Some(v) => v,
                None => return f64_round(value, places),
            },
            Some(_) => qq,
            None => return f64_round(value, places),
        }
    };
    let text = dec_str(q, -(places as i32), d.neg);
    // `str -> f64` parses are correctly rounded on both sides, exactly like
    // Python's `float(Decimal)`.
    match text.parse::<f64>() {
        Ok(v) => {
            if v == 0.0 { 0.0 } else { v }
        }
        Err(_) => 0.0,
    }
}

/// Plain f64 multiply-and-round (half away from zero). Used only when the
/// decimal string exceeds u128 range — no recursion back into callers.
/// Non-finite parses pass through (matching Python's `float(Decimal)`).
fn f64_round(value: &str, places: i64) -> f64 {
    let v: f64 = value.trim().parse().unwrap_or(0.0);
    if !v.is_finite() {
        return v;
    }
    let factor = 10f64.powi(places as i32);
    let scaled = v * factor;
    if !scaled.is_finite() {
        // Huge magnitude: rounding to `places` decimals cannot change the
        // f64 value (ulp >> quantum), so quantize is the identity —
        // exactly like Python's `float(quantized huge Decimal)`.
        return v;
    }
    let cleaned = scaled.round() / factor;
    if cleaned == 0.0 { 0.0 } else { cleaned }
}

/// Stable, bounded-precision float for generated Python source.
///
/// Mirrors `clean_generated_number(value, decimal_places)`: the float is
/// rendered with the shortest round-trip representation — the same digits
/// Python's `str(float)` feeds to `Decimal` — then quantized exactly.
pub fn clean_number(value: f64, decimal_places: i64) -> f64 {
    if !value.is_finite() {
        // Python: Decimal(str(inf/nan)) quantizes to itself; float() passes
        // it through (`inf == 0` is false, so no zero-mapping).
        return value;
    }
    // `:?` (Debug) renders the shortest round-trip with scientific notation
    // for huge magnitudes — the same digits Python's `str(float)` feeds to
    // `Decimal`. (`{}`/Display would expand 1e300 into 301 digits and
    // overflow the u128 parse.)
    clean_decimal_str(&format!("{value:?}"), decimal_places)
}

/// User-facing number without scientific notation or zero padding.
/// Mirrors `format_decimal`.
pub fn format_decimal(value: f64, decimal_places: i64) -> String {
    let places = min(18, max(0, decimal_places)) as usize;
    let cleaned = clean_number(value, decimal_places);
    if cleaned.is_nan() {
        // Python renders float nan as "nan"; Rust renders "NaN".
        return "nan".to_string();
    }
    // Format the *cleaned float* with fixed places, like Python's
    // `format(cleaned, f".{places}f")`.
    let mut text = format!("{cleaned:.places$}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    if text.is_empty() || text == "-0" {
        "0".to_string()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_str_matches_cpython_notation() {
        // Plain vs scientific switching (CPython __str__ rule).
        assert_eq!(to_decimal_str("0.000001"), "0.000001");
        assert_eq!(to_decimal_str("0.0000001"), "1E-7");
        assert_eq!(to_decimal_str("1.2300"), "1.2300");
        assert_eq!(to_decimal_str("1E-7"), "1E-7");
        assert_eq!(to_decimal_str("1E+2"), "1E+2");
        assert_eq!(to_decimal_str("100"), "100");
        assert_eq!(to_decimal_str("100.10"), "100.10");
        assert_eq!(to_decimal_str("-0"), "-0");
        assert_eq!(to_decimal_str("abc"), "0");
    }

    #[test]
    fn floor_absorbs_float_edge_noise() {
        // 0.0001 arriving as 0.00009999999999999994 must still floor to 0.0001.
        assert_eq!(floor_to_step_str("0.00009999999999999994", "0.0001"), "0.0001");
    }

    #[test]
    fn floor_normal_cases() {
        assert_eq!(floor_to_step_str("1.234567", "0.01"), "1.23");
        assert_eq!(floor_to_step_str("5", "2"), "4");
        assert_eq!(floor_to_step_str("0", "0.01"), "0");
        assert_eq!(floor_to_step_str("-3", "0.5"), "0");
        assert_eq!(floor_to_step_str("1.5", "0"), "1.5");
    }

    #[test]
    fn floor_below_step_keeps_increment_exponent() {
        // Python returns Decimal("0.0000"), not Decimal("0"): str is "0.0000".
        assert_eq!(floor_to_step_str("0.00009", "0.0001"), "0.0000");
        assert_eq!(floor_to_step_str("0.1", "0.3"), "0.0");
    }

    #[test]
    fn floor_scientific_and_rounding() {
        // 10 * 1E-8 renders "1E-7", exactly like CPython.
        assert_eq!(floor_to_step_str("1e-7", "1e-8"), "1E-7");
        assert_eq!(floor_to_step_str("123.456", "0.5"), "123.0");
        assert_eq!(floor_to_step_str("10", "3"), "9");
        assert_eq!(floor_to_step_str("100", "6"), "96");
        assert_eq!(floor_to_step_str("7", "3"), "6");
        assert_eq!(floor_to_step_str("100", "0.5"), "100");
        assert_eq!(floor_to_step_str("0.3", "0.1"), "0.3");
    }

    #[test]
    fn clean_number_quantizes_half_up() {
        assert_eq!(clean_number(1.2345678901234567, 12), 1.234567890123);
        assert_eq!(clean_number(0.0, 12), 0.0);
        assert_eq!(clean_number(2.5, 0), 3.0);
        // Exact decimal quantize: 2.675 -> 2.68 (f64 multiply-and-round
        // would give 2.67 because 2.675 is stored as 2.67499999...).
        assert_eq!(clean_decimal_str("2.675", 2).to_string(), "2.68");
        assert_eq!(clean_decimal_str("2.5", 0).to_string(), "3");
        assert_eq!(clean_decimal_str("-2.5", 0).to_string(), "-3");
        assert_eq!(clean_decimal_str("abc", 12), 0.0);
    }

    #[test]
    fn format_decimal_strips_padding() {
        assert_eq!(format_decimal(1.23000, 12), "1.23");
        assert_eq!(format_decimal(0.0, 12), "0");
        assert_eq!(format_decimal(100.0, 12), "100");
        assert!(!format_decimal(1e-7, 12).contains('e'));
    }
}
