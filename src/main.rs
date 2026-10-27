// Fixed-size chunk loops read fine as they are.
#![allow(clippy::chunks_exact_to_as_chunks, clippy::manual_checked_ops)]

mod assets;
mod biome;
mod config;
mod lod;
mod mesh;
mod model;
mod nbt;
mod region;
mod render;
mod serve;
mod textures;
mod world;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "A fast 3D web map for Minecraft worlds")]
struct Cli {
    /// Path to the config file.
    #[arg(short, long, default_value = "tealmap.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render maps once and exit.
    Render {
        /// Only render this map.
        #[arg(long)]
        map: Option<String>,
        /// Ignore saved state and render everything.
        #[arg(long)]
        full: bool,
    },
    /// Serve the web map, re-rendering as the world changes.
    Serve,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tealmap=info".into()),
        )
        .init();
    let cli = Cli::parse();
    let cfg = config::Config::load(&cli.config)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads())
        .thread_name(|i| format!("render-{i}"))
        .build()?;

    match cli.cmd {
        Cmd::Render { map, full } => {
            let maps: Vec<_> = cfg
                .maps
                .iter()
                .filter(|m| map.as_ref().is_none_or(|id| &m.id == id))
                .collect();
            if maps.is_empty() {
                bail!("no matching map in {}", cli.config.display());
            }
            for m in maps {
                let ctx = render::MapCtx::new(m.clone(), &cfg.data_dir)?;
                let up = ctx.render(&pool, full)?;
                let s = &up.stats;
                tracing::info!(
                    "{}: {} regions, {} changed chunks, {} tiles ({} empty), {:.1} MB, {} vertices, {} failed chunks, {:.2}s",
                    m.id,
                    s.regions,
                    s.chunks,
                    s.tiles,
                    s.empty_tiles,
                    s.bytes as f64 / 1e6,
                    s.vertices,
                    s.failed_chunks,
                    s.seconds
                );
                let unknown = ctx.reg.unknown();
                if !unknown.is_empty() {
                    let names: Vec<&str> =
                        unknown.iter().take(12).map(|(n, _)| n.as_str()).collect();
                    tracing::info!(
                        "{}: {} block states, {} blocks without models (shown as fallback cubes): {}{}",
                        m.id,
                        ctx.reg.len(),
                        unknown.len(),
                        names.join(", "),
                        if unknown.len() > 12 { ", …" } else { "" }
                    );
                }
            }
        }
        Cmd::Serve => serve::run(cfg, pool)?,
    }
    Ok(())
}
