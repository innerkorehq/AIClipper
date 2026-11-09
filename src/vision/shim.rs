//! Rust-facing wrapper around the Vision shim dylib (`libVisionShim`).
//!
//! The shim is compiled by `build.rs` from `vision_shim/` and dlopen'd here.
//! Normalized geometry only — top-left origin, [0,1] relative to the frame.

use anyhow::{Context, Result};
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceRect {
    /// Normalized [0,1], top-left origin.
    pub bbox: crate::types::BBox,
    pub confidence: f32,
    /// Mean colour saturation inside the box (0..1); negative when unknown (an older
    /// cache, or a shim without the probe). A printed black-and-white photo of a face
    /// reads ~0.1, a living face ~0.5.
    #[serde(default = "unknown_saturation")]
    pub saturation: f32,
}

fn unknown_saturation() -> f32 {
    -1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaliencyGrid {
    pub w: usize,
    pub h: usize,
    /// Row-major 0..1 intensities.
    pub cells: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouthLandmarks {
    pub confidence: f32,
    /// Normalized (x, y) points, top-left origin.
    pub points: Vec<(f32, f32)>,
}

type FacesFn = unsafe extern "C" fn(*const std::os::raw::c_char, *mut f32, i32) -> i32;
type SaliencyFn = unsafe extern "C" fn(*const std::os::raw::c_char, *mut f32, *mut i32, *mut i32) -> i32;
type LandmarksFn = unsafe extern "C" fn(*const std::os::raw::c_char, *mut f32, i32) -> i32;
type SaturationFn = unsafe extern "C" fn(*const std::os::raw::c_char, *const f32, i32, *mut f32) -> i32;
type VersionFn = unsafe extern "C" fn() -> i32;

#[allow(dead_code)]
pub struct VisionShim {
    _lib: &'static Library,
    faces: Symbol<'static, FacesFn>,
    saliency_attention: Symbol<'static, SaliencyFn>,
    saliency_objectness: Symbol<'static, SaliencyFn>,
    landmarks: Symbol<'static, LandmarksFn>,
    _version: Symbol<'static, VersionFn>,
    /// Absent in a shim built before the probe existed.
    saturation: Option<Symbol<'static, SaturationFn>>,
}

impl VisionShim {
    /// Resolve the dylib path: env override → compile-time build.rs path → repo default.
    pub fn resolve_path(cfg_default: &Path) -> PathBuf {
        if let Ok(p) = std::env::var("AICLIPPER_VISION_SHIM") {
            return PathBuf::from(p);
        }
        if let Some(p) = option_env!("VISION_SHIM_PATH") {
            return PathBuf::from(p);
        }
        cfg_default.to_path_buf()
    }

    pub fn load(path: &Path) -> Result<Self> {
        // Leak the Library so symbols can carry a 'static lifetime.
        let lib: &'static Library =
            Box::leak(Box::new(unsafe {
                Library::new(path).with_context(|| {
                    format!(
                        "cannot dlopen Vision shim at {} (build with `cargo build` first, or set AICLIPPER_VISION_SHIM)",
                        path.display()
                    )
                })?
            }));
        Ok(Self {
            faces: unsafe { lib.get(b"shim_faces\0") }?,
            saliency_attention: unsafe { lib.get(b"shim_saliency_attention\0") }?,
            saliency_objectness: unsafe { lib.get(b"shim_saliency_objectness\0") }?,
            landmarks: unsafe { lib.get(b"shim_face_landmarks\0") }?,
            _version: unsafe { lib.get(b"shim_version\0") }?,
            saturation: unsafe { lib.get(b"shim_face_saturation\0") }.ok(),
            _lib: lib,
        })
    }

    pub fn detect_faces(&self, image: &Path) -> Result<Vec<FaceRect>> {
        let path = c_string(image)?;
        const MAX: usize = 16;
        let mut buf = vec![0f32; MAX * 5];
        let n = unsafe { (self.faces)(path.as_ptr(), buf.as_mut_ptr(), MAX as i32) };
        if n < 0 {
            anyhow::bail!("Vision shim error on {}", image.display());
        }
        let mut out = Vec::new();
        for i in 0..n as usize {
            out.push(FaceRect {
                bbox: crate::types::BBox {
                    x: buf[i * 5],
                    y: buf[i * 5 + 1],
                    w: buf[i * 5 + 2],
                    h: buf[i * 5 + 3],
                },
                confidence: buf[i * 5 + 4],
                saturation: -1.0,
            });
        }
        if let (Some(sat), false) = (&self.saturation, out.is_empty()) {
            let rects: Vec<f32> = out.iter().flat_map(|f| [f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h]).collect();
            let mut vals = vec![-1f32; out.len()];
            let n = unsafe { (sat)(path.as_ptr(), rects.as_ptr(), out.len() as i32, vals.as_mut_ptr()) };
            if n >= 0 {
                for (f, v) in out.iter_mut().zip(vals) {
                    f.saturation = v;
                }
            }
        }
        Ok(out)
    }

    pub fn saliency_attention(&self, image: &Path) -> Result<SaliencyGrid> {
        self.saliency(image, &self.saliency_attention)
    }

    #[allow(dead_code)]
    pub fn saliency_objectness(&self, image: &Path) -> Result<SaliencyGrid> {
        self.saliency(image, &self.saliency_objectness)
    }

    fn saliency(&self, image: &Path, sym: &Symbol<'static, SaliencyFn>) -> Result<SaliencyGrid> {
        let path = c_string(image)?;
        let mut out = vec![0f32; 48 * 48];
        let (mut w, mut h) = (0i32, 0i32);
        let n = unsafe { (sym)(path.as_ptr(), out.as_mut_ptr(), &mut w, &mut h) };
        if n < 0 {
            anyhow::bail!("Vision saliency error on {}", image.display());
        }
        let (w, h) = (w as usize, h as usize);
        out.truncate(w * h);
        Ok(SaliencyGrid { w, h, cells: out })
    }

    pub fn face_landmarks(&self, image: &Path) -> Result<Vec<MouthLandmarks>> {
        let path = c_string(image)?;
        const MAX: usize = 8;
        const STRIDE: usize = 45;
        let mut buf = vec![0f32; MAX * STRIDE];
        let n = unsafe { (self.landmarks)(path.as_ptr(), buf.as_mut_ptr(), MAX as i32) };
        if n < 0 {
            anyhow::bail!("Vision landmarks error on {}", image.display());
        }
        let mut out = Vec::new();
        for f in 0..n as usize {
            let mut pts = Vec::new();
            for p in 0..22 {
                let x = buf[f * STRIDE + 1 + p * 2];
                let y = buf[f * STRIDE + 2 + p * 2];
                pts.push((x, y));
            }
            out.push(MouthLandmarks { confidence: buf[f * STRIDE], points: pts });
        }
        Ok(out)
    }
}

fn c_string(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes()).context("path contains NUL byte")
}