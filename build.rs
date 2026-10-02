//! Compiles the Swift Vision shim (`vision_shim/`) into a dylib with `swiftc`
//! when building. Exposes the resulting absolute path via `VISION_SHIM_PATH`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let src = manifest.join("vision_shim/Sources/VisionShim/VisionShim.swift");
    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        // Not a source checkout — keep the build green; runtime reports a
        // clear error when Vision is actually needed.
        println!("cargo:warning=vision shim source not found at {}", src.display());
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let ext = if cfg!(target_os = "macos") { "libVisionShim.dylib" } else { "libVisionShim.so" };
    let dylib = out_dir.join(ext);

    let source_modified = src.metadata().and_then(|m| m.modified()).ok();
    let binary_modified = dylib.metadata().and_then(|m| m.modified()).ok();
    let needs_build = env::var_os("AICLIPPER_SKIP_VISION_BUILD").is_none()
        && (!dylib.exists()
            || source_modified.zip(binary_modified).map(|(s, b)| b < s).unwrap_or(true));

    if needs_build {
        println!("cargo:warning=compiling Vision shim (swiftc -O -emit-library)…");
        let mut cmd = Command::new("swiftc");
        cmd.arg("-O")
            .arg("-emit-library")
            .arg(src.as_os_str())
            .arg("-o")
            .arg(dylib.as_os_str());
        for fw in ["Vision", "Foundation", "CoreVideo", "ImageIO", "CoreGraphics"] {
            cmd.arg("-framework").arg(fw);
        }
        let out = cmd.output()?;
        if !out.status.success() {
            anyhow::bail!(
                "swiftc failed to build Vision shim:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    println!("cargo:rustc-env=VISION_SHIM_PATH={}", dylib.display());
    Ok(())
}