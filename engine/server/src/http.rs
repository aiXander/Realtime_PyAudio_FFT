//! Static file server for the UI (`server/io/http_server.py`): browsers
//! refuse ES modules from file://. `Cache-Control: no-cache` on every
//! response (so a refresh after editing ui/ never runs stale modules) and
//! forced MIME types for the extensions some OS registries get wrong.

use axum::extract::Request;
use axum::http::{HeaderValue, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use std::path::Path;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

fn forced_mime(path: &str) -> Option<&'static str> {
    let ext = path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase())?;
    Some(match ext.as_str() {
        "js" | "mjs" => "text/javascript",
        "css" => "text/css",
        "json" => "application/json",
        "wasm" => "application/wasm",
        "svg" => "image/svg+xml",
        "html" => "text/html",
        _ => return None,
    })
}

async fn force_mime(req: Request, next: Next) -> Response {
    let mime = forced_mime(req.uri().path());
    let mut resp = next.run(req).await;
    if let Some(m) = mime
        && resp.status().is_success()
    {
        resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(m));
    }
    resp
}

pub fn router(root: &Path) -> axum::Router {
    axum::Router::new()
        .fallback_service(ServeDir::new(root))
        .layer(middleware::from_fn(force_mime))
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
}
