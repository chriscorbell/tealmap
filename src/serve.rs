//! HTTP server: the embedded viewer, map files, and live updates over SSE.

use crate::config::Config;
use crate::render::{MapCtx, Update};
use anyhow::Result;
use axum::Router;
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures_util::Stream;
use std::collections::HashMap;
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
    /// Gzipped viewer assets by path, for the ones worth compressing.
    gzipped: Arc<HashMap<String, Vec<u8>>>,
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
        // Debug builds read the viewer from disk, so it can change under a running server.
        gzipped: Arc::new(if cfg!(debug_assertions) {
            HashMap::new()
        } else {
            gzip_assets(
                Viewer::iter()
                    .filter_map(|p| Some((p.to_string(), Viewer::get(&p)?.data.into_owned()))),
            )
        }),
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
            // What each map's last pass failed with, so a problem that lasts
            // is logged once rather than every poll.
            let mut failing: Vec<Option<String>> = vec![None; ctxs.len()];
            loop {
                for (ctx, failing) in ctxs.iter().zip(&mut failing) {
                    let publish = |up: &Update| {
                        if !up.hires.is_empty() || !up.lod.is_empty() {
                            let msg = serde_json::json!({ "map": ctx.cfg.id, "update": up });
                            let _ = tx.send(msg.to_string());
                        }
                    };
                    match ctx.render(&pool, false, publish) {
                        Ok(up) => {
                            if failing.take().is_some() {
                                tracing::info!("render {}: working again", ctx.cfg.id);
                            }
                            if !up.hires.is_empty() || !up.lod.is_empty() {
                                let s = &up.stats;
                                tracing::info!(
                                    "{}: {} changed chunks → {} tiles in {:.2}s",
                                    ctx.cfg.id,
                                    s.chunks,
                                    s.tiles,
                                    s.seconds
                                );
                            }
                        }
                        Err(e) => {
                            let message = format!("{e:#}");
                            if failing.as_ref() != Some(&message) {
                                tracing::error!("render {}: {message}", ctx.cfg.id);
                                *failing = Some(message);
                            }
                        }
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
        || (s.starts_with("lod/") && s.ends_with(".tmh"))
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
    // Tiles can be used from the browser's cache for a while and revalidated in
    // the background after that. Live updates ask for changed tiles past the cache.
    let cache = if path.ends_with(".tmh") || path == "textures.bin" {
        "max-age=300, stale-while-revalidate=604800"
    } else {
        "no-cache"
    };
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
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
    let gz = body.starts_with(&[0x1f, 0x8b]);
    let ctype = if path.ends_with(".json") {
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

/// Gzips the text assets once, at their smallest, since they never change
/// while the server runs and every first visit downloads them.
fn gzip_assets(files: impl Iterator<Item = (String, Vec<u8>)>) -> HashMap<String, Vec<u8>> {
    let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::best());
    files
        .filter(|(path, _)| {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            let kind = mime.essence_str();
            kind.starts_with("text/")
                || kind.ends_with("javascript")
                || kind.ends_with("json")
                || kind.ends_with("+xml")
        })
        .filter_map(|(path, data)| {
            let mut gz = vec![0; compressor.gzip_compress_bound(data.len())];
            let n = compressor.gzip_compress(&data, &mut gz).ok()?;
            gz.truncate(n);
            (n < data.len()).then_some((path, gz))
        })
        .collect()
}

/// Whether the client takes gzip, per its Accept-Encoding header. A wildcard
/// only speaks for codings the header doesn't name.
fn accepts_gzip(headers: &HeaderMap) -> bool {
    let Some(accepted) = headers
        .get(header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let mut wildcard = false;
    for coding in accepted.split(',') {
        let mut parts = coding.split(';').map(str::trim);
        let name = parts.next().unwrap_or("");
        let refused = parts.any(|p| {
            p.split_once('=').is_some_and(|(k, q)| {
                k.trim().eq_ignore_ascii_case("q")
                    && q.trim().parse::<f32>().is_ok_and(|q| q == 0.0)
            })
        });
        if name.eq_ignore_ascii_case("gzip") {
            return !refused;
        }
        wildcard |= name == "*" && !refused;
    }
    wildcard
}

async fn viewer(State(s): State<AppState>, uri: Uri, req_headers: HeaderMap) -> Response {
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
    if let Some(gz) = s.gzipped.get(path) {
        h.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
        if accepts_gzip(&req_headers) {
            h.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
            return (h, gz.clone()).into_response();
        }
    }
    (h, file.data.into_owned()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gzip_negotiation() {
        let accepts = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert(header::ACCEPT_ENCODING, HeaderValue::from_str(v).unwrap());
            accepts_gzip(&h)
        };
        assert!(accepts("gzip"));
        assert!(accepts("gzip, deflate, br, zstd"));
        assert!(accepts("br;q=1.0, GZIP;q=0.5"));
        assert!(accepts("*"));
        assert!(!accepts("br"));
        assert!(!accepts("identity"));
        assert!(!accepts("gzip;q=0"));
        assert!(!accepts("gzip; q=0.0, br"));
        assert!(!accepts("gzip;Q=0"));
        assert!(!accepts("gzip;q=0, *;q=1"));
        assert!(!accepts("*;q=1, gzip;q=0"));
        assert!(!accepts("*;q=0"));
        assert!(!accepts_gzip(&HeaderMap::new()));
    }

    #[test]
    fn only_text_that_shrinks_is_gzipped() {
        let text = b"const tealmap = 1;\n".repeat(200);
        let gz = gzip_assets(
            [
                ("assets/a.js", text.clone()),
                ("assets/a.css", text.clone()),
                ("index.html", text.clone()),
                ("favicon.svg", text.clone()),
                ("x.png", text.clone()),
                ("tiny.css", b"a".to_vec()),
            ]
            .into_iter()
            .map(|(p, d)| (p.to_string(), d)),
        );
        let mut names: Vec<_> = gz.keys().map(String::as_str).collect();
        names.sort();
        assert_eq!(
            names,
            ["assets/a.css", "assets/a.js", "favicon.svg", "index.html"]
        );
        let mut out = Vec::new();
        std::io::Read::read_to_end(
            &mut flate2::read::GzDecoder::new(&gz["assets/a.js"][..]),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, text);
    }

    #[test]
    fn path_allowlist() {
        assert!(allowed(Path::new("map.json")));
        assert!(allowed(Path::new("hires/0_0/1_2.tmh")));
        assert!(allowed(Path::new("lod/3/-1_0.tmh")));
        assert!(!allowed(Path::new("lod/src1/0_0.bin")));
        assert!(!allowed(Path::new("../config.toml")));
        assert!(!allowed(Path::new("hires/../../x.tmh")));
        assert!(!allowed(Path::new("state/0_0.bin")));
        assert!(!allowed(Path::new("/etc/passwd")));
    }
}
