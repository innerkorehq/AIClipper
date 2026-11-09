//! Vision service: frame extraction + cached Vision detections, on top of the
//! shim dylib. Used by shot classification (§5.6), active-speaker resolution
//! (§5.7), and the trajectory builder (§5.8). Degrades gracefully to a
//! no-Vision mode with a clear error if the shim cannot be loaded.

pub mod decoys;
pub mod shim;

pub use shim::{FaceRect, MouthLandmarks, SaliencyGrid, VisionShim};

use crate::config::Config;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct VisionService {
    shim: Option<Arc<VisionShim>>,
    cfg: Arc<Config>,
    frames_dir: PathBuf,
}

/// A per-frame sample of where the interesting content is.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FrameSample {
    pub frame_idx: u32,
    pub time: f64,
    /// Detected faces (any).
    pub faces: Vec<FaceRect>,
    /// Attention saliency grid if computed.
    pub saliency: Option<SaliencyGrid>,
}

impl VisionService {
    pub fn new(cfg: &Config) -> (Self, bool) {
        let path = VisionShim::resolve_path(&cfg.vision_shim);
        match VisionShim::load(&path) {
            Ok(shim) => {
                tracing::info!("Vision shim loaded from {}", path.display());
                let svc = Self {
                    shim: Some(Arc::new(shim)),
                    cfg: Arc::new(cfg.clone()),
                    frames_dir: cfg.frames_dir(),
                };
                (svc.clone(), true)
            }
            Err(e) => {
                tracing::warn!("Vision shim unavailable: {e:#}");
                let svc = Self { shim: None, cfg: Arc::new(cfg.clone()), frames_dir: cfg.frames_dir() };
                (svc, false)
            }
        }
    }

    pub fn available(&self) -> bool {
        self.shim.is_some()
    }

    /// Extract a single frame at `time` seconds into a JPEG file (cached).
    pub async fn frame_at(&self, video: &Path, time: f64) -> Result<PathBuf> {
        let stem = file_stem(video);
        let path = self.frames_dir.join(format!("{stem}_t{time:.3}.jpg"));
        if path.exists() {
            return Ok(path);
        }
        let mut cmd = tokio::process::Command::new(&self.cfg.ffmpeg);
        cmd.arg("-y")
            .arg("-ss")
            .arg(time.to_string())
            .arg("-i")
            .arg(video.as_os_str())
            .arg("-frames:v")
            .arg("1")
            .arg("-vf")
            .arg("scale='trunc(iw/2)*2:trunc(ih/2)*2'")
            .arg("-q:v")
            .arg("3")
            .arg(path.as_os_str());
        crate::util::run_ok(&mut cmd, &format!("frame extract t={time:.2}s")).await?;
        Ok(path)
    }

    /// Face detections for a list of times (cached per frame).
    pub async fn faces_at(&self, video: &Path, times: &[f64]) -> Result<Vec<Vec<FaceRect>>> {
        let stem = file_stem(video);
        let mut work = Vec::with_capacity(times.len());
        for t in times {
            let frame = self.frame_at(video, *t).await?;
            work.push((frame, format!("{stem}_t{t:.3}_faces.json")));
        }
        self.blocking_map(work, |shim, p| shim.detect_faces(p)).await
    }

    /// Attention saliency grids for a list of times (cached per frame).
    pub async fn saliency_at(&self, video: &Path, times: &[f64]) -> Result<Vec<Option<SaliencyGrid>>> {
        let stem = file_stem(video);
        let mut work = Vec::with_capacity(times.len());
        for t in times {
            let frame = self.frame_at(video, *t).await?;
            work.push((frame, format!("{stem}_t{t:.3}_saliency.json")));
        }
        self.blocking_map_opt(work, |shim, p| shim.saliency_attention(p)).await
    }

    /// Mouth landmarks for a list of times (cached per frame).
    pub async fn landmarks_at(&self, video: &Path, times: &[f64]) -> Result<Vec<Vec<MouthLandmarks>>> {
        let stem = file_stem(video);
        let mut work = Vec::with_capacity(times.len());
        for t in times {
            let frame = self.frame_at(video, *t).await?;
            work.push((frame, format!("{stem}_t{t:.3}_landmarks.json")));
        }
        self.blocking_map(work, |shim, p| shim.face_landmarks(p)).await
    }

    /// Run a per-frame shim call inside spawn_blocking, with a JSON cache.
    async fn blocking_map<T, F>(
        &self,
        work: Vec<(PathBuf, String)>,
        compute: F,
    ) -> Result<Vec<T>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
        F: Fn(&VisionShim, &Path) -> Result<T> + Send + Sync + 'static,
    {
        let frames_dir = self.frames_dir.clone();
        let shim = self.shim.clone().context("Vision shim unavailable")?;
        let compute = std::sync::Arc::new(compute);
        let handles: Vec<_> = work
            .into_iter()
            .map(|(frame, key)| {
                let shim = shim.clone();
                let compute = compute.clone();
                let frames_dir = frames_dir.clone();
                tokio::task::spawn_blocking(move || {
                    let cache = frames_dir.join(&key);
                    if let Ok(s) = std::fs::read_to_string(&cache) {
                        if let Ok(v) = serde_json::from_str(&s) {
                            return Ok(v);
                        }
                    }
                    let value = compute(shim.as_ref(), &frame);
                    if let Ok(v) = &value {
                        if let Ok(bytes) = serde_json::to_vec(v) {
                            let _ = std::fs::write(&cache, bytes);
                        }
                    }
                    value
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.await.context("vision worker join")??);
        }
        Ok(out)
    }

    /// Like `blocking_map` but failures return `None` entries instead of failing the stage.
    async fn blocking_map_opt<T, F>(
        &self,
        work: Vec<(PathBuf, String)>,
        compute: F,
    ) -> Result<Vec<Option<T>>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
        F: Fn(&VisionShim, &Path) -> Result<T> + Send + Sync + 'static,
    {
        let frames_dir = self.frames_dir.clone();
        let shim = self.shim.clone().context("Vision shim unavailable")?;
        let compute = std::sync::Arc::new(compute);
        let handles: Vec<_> = work
            .into_iter()
            .map(|(frame, key)| {
                let shim = shim.clone();
                let compute = compute.clone();
                let frames_dir = frames_dir.clone();
                tokio::task::spawn_blocking(move || -> anyhow::Result<Option<T>> {
                        let cache = frames_dir.join(&key);
                        if cache.exists() {
                            if let Ok(s) = std::fs::read_to_string(&cache) {
                                if let Ok(serde_json::Value::Null) =
                                    serde_json::from_str::<serde_json::Value>(&s)
                                {
                                    return Ok(None);
                                }
                                if let Ok(v) = serde_json::from_str(&s) {
                                    return Ok(Some(v));
                                }
                            }
                        }
                        match compute(shim.as_ref(), &frame) {
                        Ok(v) => {
                            let _ = std::fs::write(&cache, serde_json::to_vec(&v).unwrap_or_default());
                            Ok(Some(v))
                        }
                        Err(_) => {
                            let _ = std::fs::write(&cache, b"null");
                            Ok(None)
                        }
                    }
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.await.context("vision worker join")??);
        }
        Ok(out)
    }

    /// Return a deep clone usable inside blocking tasks (cheap).
    #[allow(dead_code)]
    pub fn clone_for_blocking(&self) -> Self {
        self.clone()
    }
}

fn file_stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("vid").to_owned()
}

/// Compute a concentration score from a saliency grid. Lower score = more
/// concentrated (a single point of interest).
pub fn saliency_concentration(grid: &SaliencyGrid) -> f64 {
    if grid.cells.iter().all(|c| *c <= 0.0) {
        return 1.0; // flat → fully diffuse
    }
    let total: f32 = grid.cells.iter().sum();
    let mut entropy = 0.0f64;
    let mut max = 0.0f32;
    for &c in &grid.cells {
        if c <= 0.0 {
            continue;
        }
        let p = f64::from(c / total);
        entropy -= p * p.ln();
        max = max.max(c);
    }
    let n = grid.cells.len() as f64;
    let max_entropy = if n > 1.0 { n.ln() } else { 1.0 };
    let norm_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 1.0 };
    let _ = max;
    norm_entropy
}

/// Extrude a target bbox (normalized) from a saliency grid: centroid + extent.
/// Used for the `saliency_crop` trajectory target per frame.
pub fn saliency_target_bbox(grid: &SaliencyGrid, deflect_x: f32, deflect_y: f32) -> crate::types::BBox {
    let (gw, gh) = (grid.w.max(1), grid.h.max(1));
    let mut sum = 0.0f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for y in 0..gh {
        for x in 0..gw {
            let v = grid.cells[y * gw + x];
            if v <= 0.0 {
                continue;
            }
            sum += v;
            cx += v * x as f32;
            cy += v * y as f32;
        }
    }
    if sum <= 0.0 {
        return crate::types::BBox { x: deflect_x, y: deflect_y, w: 0.5, h: 0.5 };
    }
    cx /= sum * gw as f32;
    cy /= sum * gh as f32;
    // effective radius from weighted std-dev of the mass
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for y in 0..gh {
        for x in 0..gw {
            let v = grid.cells[y * gw + x];
            if v <= 0.0 {
                continue;
            }
            vx += v * (x as f32 / gw as f32 - cx).powi(2);
            vy += v * (y as f32 / gh as f32 - cy).powi(2);
        }
    }
    let rx = (vx / sum).max(0.001).sqrt() * 2.0;
    let ry = (vy / sum).max(0.001).sqrt() * 2.0;
    let w = (rx * 3.0).clamp(0.25, 1.0);
    let h = (ry * 3.0).clamp(0.3, 1.0);
    crate::types::BBox {
        x: (cx - w / 2.0).clamp(0.0, 1.0 - w),
        y: (cy - h / 2.0).clamp(0.0, 1.0 - h),
        w,
        h,
    }
}