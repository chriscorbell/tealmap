//! HTTP server: the embedded viewer, map files, and live updates over SSE.

use crate::config::Config;
use crate::render::MapCtx;
use anyhow::Result;
use axum::Router;
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures_util::Stream;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

#[derive(rust_embed::RustEmbed)]
#[folder = "viewer/dist"]
struct Viewer;

#[derive(Clone)]
struct AppState {
    maps_dir: PathBuf,
    maps: Arc<Vec<serde_json::Value>>,
    events: broadcast::Sender<String>,
}

pub fn run(cfg: Config, pool: rayon::ThreadPool) -> Result<()> {
    let (tx, _) = broadcast::channel::<String>(64);
    let maps: Vec<serde_json::Value> = cfg
        .maps
        .iter()
        .map(|m| serde_json::json!({ "id": m.id, "name": m.display_name() }))
        .collect();
    let state = AppState {
        maps_dir: cfg.data_dir.join("maps"),
        maps: Arc::new(maps),
        events: tx.clone(),
    };

    // Render loop on its own thread; rendering itself runs on the rayon pool.
    let render_cfg = cfg.clone();
    std::thread::Builder::new()
        .name("watch".into())
        .spawn(move || {
            let ctxs: Vec<MapCtx> = render_cfg
                .maps
                .iter()
                .filter_map(|m| match MapCtx::new(m.clone(), &render_cfg.data_dir) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::error!("map {}: {e:#}", m.id);
                        None
                    }
                })
                .collect();
            loop {
                for ctx in &ctxs {
                    match ctx.render(&pool, false) {
                        Ok(up) => {
                            if !up.hires.is_empty() || !up.lowres.is_empty() {
                                let s = &up.stats;
                                tracing::info!(
                                    "{}: {} changed chunks → {} tiles in {:.2}s",
                                    ctx.cfg.id,
                                    s.chunks,
                                    s.tiles,
                                    s.seconds
                                );
                                let msg = serde_json::json!({ "map": ctx.cfg.id, "update": up });
                                let _ = tx.send(msg.to_string());
                            }
                        }
                        Err(e) => tracing::error!("render {}: {e:#}", ctx.cfg.id),
                    }
                }
                std::thread::sleep(Duration::from_secs(render_cfg.poll_seconds.max(1)));
            }
        })?;

    let app = Router::new()
        .route("/api/maps", get(list_maps))
        .route("/api/events", get(events))
        .route("/maps/{id}/{*path}", get(map_file))
        .fallback(viewer)
        .with_state(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&cfg.listen).await?;
        tracing::info!("serving on http://{}", cfg.listen);
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

async fn list_maps(State(s): State<AppState>) -> impl IntoResponse {
    let mut h = HeaderMap::new();
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    (h, axum::Json(serde_json::Value::Array((*s.maps).clone())))
}

async fn events(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = BroadcastStream::new(s.events.subscribe())
        .filter_map(|m| m.ok())
        .map(|m| Ok(Event::default().event("update").data(m)));
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(20)))
}

/// Only these files under a map directory are served.
fn allowed(rel: &Path) -> bool {
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return false;
    }
    let s = rel.to_string_lossy();
    s == "map.json"
        || s == "textures.bin"
        || (s.starts_with("hires/") && s.ends_with(".tmh"))
        || (s.starts_with("lowres/") && s.ends_with(".png"))
}

async fn map_file(
    State(s): State<AppState>,
    UrlPath((id, path)): UrlPath<(String, String)>,
    req_headers: HeaderMap,
) -> Response {
    let rel = PathBuf::from(&path);
    let id_ok = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !id_ok || !allowed(&rel) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let full = s.maps_dir.join(&id).join(&rel);
    let Ok(meta) = tokio::fs::metadata(&full).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let etag = format!("\"{:x}-{:x}\"", mtime, meta.len());
    let mut h = HeaderMap::new();
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
    if req_headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        == Some(etag.as_str())
    {
        return (StatusCode::NOT_MODIFIED, h).into_response();
    }
    let Ok(body) = tokio::fs::read(&full).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let gz = path.ends_with(".tmh") || path == "textures.bin";
    let ctype = if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    };
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(ctype));
    if gz {
        h.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
    (h, body).into_response()
}

async fn viewer(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let (file, path) = match Viewer::get(path) {
        Some(f) => (f, path),
        None => match Viewer::get("index.html") {
            Some(f) => (f, "index.html"),
            None => return (StatusCode::NOT_FOUND, "viewer not built").into_response(),
        },
    };
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let mut h = HeaderMap::new();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime.as_ref()).unwrap(),
    );
    // Vite puts content hashes in asset names, so those can be cached forever.
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    (h, file.data.into_owned()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_allowlist() {
        assert!(allowed(Path::new("map.json")));
        assert!(allowed(Path::new("hires/0_0/1_2.tmh")));
        assert!(allowed(Path::new("lowres/1/-1_0.png")));
        assert!(!allowed(Path::new("../config.toml")));
        assert!(!allowed(Path::new("hires/../../x.tmh")));
        assert!(!allowed(Path::new("state/0_0.bin")));
        assert!(!allowed(Path::new("/etc/passwd")));
    }
}
