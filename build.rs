// The server embeds viewer/dist. Make sure the folder exists so the Rust
// crate builds (and tests run) before the viewer has been built.
fn main() {
    let dist = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("viewer/dist");
    if !dist.exists() {
        std::fs::create_dir_all(&dist).unwrap();
    }
    println!("cargo:rerun-if-changed=viewer/dist");
}
