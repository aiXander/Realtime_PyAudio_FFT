//! Python-semantics helpers over `serde_json::Value`: the reference server
//! coerces config / message values with `bool()` / `int()` / `float()` /
//! `str()` and formats errors with `repr()`. These reproduce what matters.

use serde_json::Value;

/// `dict.get(key)`: a missing key and an explicit `null` both read as None.
pub fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v.get(key) {
        None | Some(Value::Null) => None,
        Some(x) => Some(x),
    }
}

/// Python truthiness (`bool(x)`).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `bool(msg.get(key, default))`: an explicit `null` is falsy.
pub fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    match v.get(key) {
        None => default,
        Some(x) => truthy(x),
    }
}

/// `isinstance(x, int) and not isinstance(x, bool)`.
pub fn as_int(v: &Value) -> Option<i128> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from)),
        _ => None,
    }
}

/// `isinstance(x, (int, float)) and not isinstance(x, bool)`.
pub fn is_number(v: &Value) -> bool {
    matches!(v, Value::Number(_))
}

/// Why a Python-style coercion failed: `TypeError` (the reference treats it
/// as a malformed config) or `ValueError` (an invalid value).
#[derive(Debug)]
pub enum CoerceError {
    Type(String),
    Value(String),
}

/// `int(x)`.
pub fn py_int(v: &Value) -> Result<i64, CoerceError> {
    match v {
        Value::Bool(b) => Ok(*b as i64),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                i64::try_from(u).map_err(|_| CoerceError::Value(format!("int too large: {u}")))
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_finite() {
                    Ok(f.trunc() as i64)
                } else {
                    Err(CoerceError::Value(format!("cannot convert float {f} to integer")))
                }
            }
        }
        Value::String(s) => s.trim().replace('_', "").parse::<i64>().map_err(|_| {
            CoerceError::Value(format!("invalid literal for int() with base 10: {}", repr_str(s)))
        }),
        other => Err(CoerceError::Type(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            type_name(other)
        ))),
    }
}

/// `float(x)`.
pub fn py_float(v: &Value) -> Result<f64, CoerceError> {
    match v {
        Value::Bool(b) => Ok(*b as i64 as f64),
        Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => {
            let t = s.trim().to_ascii_lowercase();
            let parsed = match t.as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
                "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
                "nan" | "+nan" | "-nan" => Some(f64::NAN),
                _ => t.replace('_', "").parse::<f64>().ok(),
            };
            parsed.ok_or_else(|| {
                CoerceError::Value(format!("could not convert string to float: {}", repr_str(s)))
            })
        }
        other => Err(CoerceError::Type(format!(
            "float() argument must be a string or a real number, not '{}'",
            type_name(other)
        ))),
    }
}

/// `str(x)`.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => repr(other),
    }
}

/// Python `type(x).__name__` of the JSON value.
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `repr(str)`: single quotes unless the string contains one and no double.
pub fn repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// Python `repr(float)`.
pub fn repr_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == f.trunc() && f.abs() < 1e16 {
        return format!("{f:.1}");
    }
    let s = format!("{f:e}");
    let exp: i32 = s.split('e').nth(1).and_then(|e| e.parse().ok()).unwrap_or(0);
    if (-4..16).contains(&exp) {
        format!("{f}")
    } else {
        let (m, _) = s.split_once('e').unwrap_or((&s, "0"));
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", exp.abs())
    }
}

/// Python `repr()` of a JSON value.
pub fn repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => {
            if n.is_f64() {
                repr_float(n.as_f64().unwrap_or(f64::NAN))
            } else {
                n.to_string()
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(o) => {
            let items: Vec<String> = o
                .iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), repr(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// `f"{x:g}"` for the small set of bounds the validators print.
pub fn fmt_g(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coercions() {
        assert!(!truthy(&json!(0)));
        assert!(truthy(&json!("x")));
        assert!(!truthy(&json!({})));
        assert_eq!(py_int(&json!(2.9)).unwrap(), 2);
        assert_eq!(py_int(&json!("12")).unwrap(), 12);
        assert!(matches!(py_int(&Value::Null), Err(CoerceError::Type(_))));
        assert!(matches!(py_int(&json!("x")), Err(CoerceError::Value(_))));
        assert_eq!(py_float(&json!("1.5")).unwrap(), 1.5);
        assert_eq!(py_str(&json!(9000)), "9000");
        assert_eq!(repr(&json!("no_such")), "'no_such'");
        assert_eq!(repr(&Value::Null), "None");
        assert_eq!(repr(&json!(["a", 1])), "['a', 1]");
        assert_eq!(repr_float(1e-5), "1e-05");
        assert_eq!(repr_float(0.25), "0.25");
        assert_eq!(repr_float(30.0), "30.0");
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(1.0), "1");
    }
}
