use crate::resources::gaussians::{Gaussian3d, Gaussians};
use anyhow::{Context, bail, ensure};
use image::ImageReader;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use zip::ZipArchive;

struct SogArchive {
    files: HashMap<String, Vec<u8>>,
}

impl SogArchive {
    fn from_zip_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let cursor = Cursor::new(bytes);
        let mut zip = ZipArchive::new(cursor).context("failed to open .sog as zip archive")?;

        let mut files = HashMap::new();

        for i in 0..zip.len() {
            let mut file = zip
                .by_index(i)
                .with_context(|| format!("failed to read zip entry at index {i}"))?;

            if file.is_dir() {
                continue;
            }

            let file_name = file.name().to_string();
            let mut data = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut data)
                .with_context(|| format!("failed to read zip entry: {file_name}"))?;

            files.insert(file_name, data);
        }

        Ok(Self { files })
    }

    pub fn get(&self, file_name: &str) -> anyhow::Result<&[u8]> {
        self.files
            .get(file_name)
            .map(|v| v.as_slice())
            .with_context(|| format!("missing SOG archive entry: {file_name}"))
    }

    pub fn read_json<T>(&self, name: &str) -> anyhow::Result<T>
    where
        T: DeserializeOwned,
    {
        let bytes = self.get(name)?;

        serde_json::from_slice(bytes).with_context(|| format!("failed to parse json entry: {name}"))
    }

    pub fn read_webp_rgb(&self, name: &str) -> anyhow::Result<ImageData> {
        let bytes = self.get(name)?;

        let image = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .with_context(|| format!("failed to guess image format: {name}"))?
            .decode()
            .with_context(|| format!("failed to decode WebP image: {name}"))?;

        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();

        Ok(ImageData {
            width,
            height,
            data: rgb.into_raw(),
        })
    }

    pub fn read_webp_rgba(&self, name: &str) -> anyhow::Result<ImageData> {
        let bytes = self.get(name)?;

        let image = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .with_context(|| format!("failed to guess image format: {name}"))?
            .decode()
            .with_context(|| format!("failed to decode WebP image: {name}"))?;

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();

        Ok(ImageData {
            width,
            height,
            data: rgba.into_raw(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct SogMeta {
    version: u32,

    #[serde(default)]
    pub asset: Option<SogAssetMeta>,

    pub count: usize,

    #[serde(default)]
    pub antialias: bool,

    pub means: SogMeansMeta,
    pub scales: SogScalesMeta,
    pub quats: SogQuatsMeta,
    pub sh0: SogSh0Meta,

    #[serde(rename = "shN")]
    #[serde(default)]
    pub shn: Option<SogShNMeta>,
}

#[derive(Debug, Deserialize)]
pub struct SogAssetMeta {
    #[serde(default)]
    pub generator: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SogMeansMeta {
    pub mins: [f32; 3],
    pub maxs: [f32; 3],
    pub files: [String; 2],
}

#[derive(Debug, Deserialize)]
pub struct SogScalesMeta {
    pub codebook: Vec<f32>,
    pub files: [String; 1],
}

#[derive(Debug, Deserialize)]
pub struct SogQuatsMeta {
    pub files: [String; 1],
}

#[derive(Debug, Deserialize)]
pub struct SogSh0Meta {
    pub codebook: Vec<f32>,
    pub files: [String; 1],
}

#[derive(Debug, Deserialize)]
pub struct SogShNMeta {
    pub count: usize,
    pub bands: u32,
    pub codebook: Vec<f32>,
    pub files: [String; 2],
}

#[derive(Debug, Clone)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ImageData {
    pub fn pixel_rgb(&self, index: usize) -> anyhow::Result<[u8; 3]> {
        let base = index * 3; // rgb

        ensure!(
            base + 2 < self.data.len(),
            "RGB pixel index out of bounds: index={}, pixels={}",
            index,
            self.width as usize * self.height as usize
        );

        Ok([self.data[base], self.data[base + 1], self.data[base + 2]])
    }
    pub fn pixel_rgba(&self, index: usize) -> anyhow::Result<[u8; 4]> {
        let base = index * 4; // rgba

        ensure!(
            base + 3 < self.data.len(),
            "RGBA pixel index out of bounds: index={}, pixels={}",
            index,
            self.width as usize * self.height as usize
        );

        Ok([
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
            self.data[base + 3],
        ])
    }
}

pub fn load_sog_from_bytes(bytes: &[u8]) -> anyhow::Result<Gaussians> {
    let archive = SogArchive::from_zip_bytes(bytes)?;

    let meta: SogMeta = archive.read_json("meta.json")?;
    ensure_supported_sog_format(&meta)?;

    let mut gaussians = Vec::<Gaussian3d>::with_capacity(meta.count);

    struct ShNImages {
        centroids: ImageData,
        labels: ImageData,
    }

    // read webp
    let means_l = archive.read_webp_rgb(&meta.means.files[0])?;
    let means_u = archive.read_webp_rgb(&meta.means.files[1])?;
    let quats = archive.read_webp_rgba(&meta.quats.files[0])?;
    let scales = archive.read_webp_rgb(&meta.scales.files[0])?;
    let sh0 = archive.read_webp_rgba(&meta.sh0.files[0])?;
    let shn_images: Option<_> = if let Some(shn) = &meta.shn {
        let centroids = archive.read_webp_rgb(&shn.files[0])?;
        let labels = archive.read_webp_rgb(&shn.files[1])?;

        Some(ShNImages { centroids, labels })
    } else {
        None
    };

    for i in 0..meta.count {
        // decode position
        let means_l_rgb = means_l.pixel_rgb(i)?;
        let means_u_rgb = means_u.pixel_rgb(i)?;
        let position = decode_position(means_l_rgb, means_u_rgb, meta.means.mins, meta.means.maxs);

        // decode rotation
        let quat_rgba = quats.pixel_rgba(i)?;
        let rotation = decode_orientation(quat_rgba)?;

        // decode scale
        let scale_rgb = scales.pixel_rgb(i)?;
        let scale = [
            meta.scales.codebook[scale_rgb[0] as usize],
            meta.scales.codebook[scale_rgb[1] as usize],
            meta.scales.codebook[scale_rgb[2] as usize],
        ];

        // decode base color / opacity
        let mut sh = [0.0; 48];
        let sh0_rgba = sh0.pixel_rgba(i)?;
        sh[0] = meta.sh0.codebook[sh0_rgba[0] as usize];
        sh[1] = meta.sh0.codebook[sh0_rgba[1] as usize];
        sh[2] = meta.sh0.codebook[sh0_rgba[2] as usize];

        // SOG stores opacity as alpha in [0, 1], while our renderer expects
        // the raw 3DGS opacity value and applies sigmoid(opacity) in the preprocess pass.
        // Convert alpha back to logit space so SOG and PLY share the same internal representation.
        let opacity = alpha_to_raw_opacity(sh0_rgba[3] as f32 / 255.0);

        // decode rest sh
        if let Some(shn_images) = &shn_images {
            let shn = meta.shn.as_ref().unwrap();
            let coeff_count = shn_coeff_count(shn.bands)?;
            let label_rgb = shn_images.labels.pixel_rgb(i)?;
            let label = decode_shn_label(label_rgb);

            for c in 0..coeff_count {
                let rgb_coeff = decode_shn_coeff_rgb(
                    label,
                    c,
                    coeff_count,
                    &shn_images.centroids,
                    &shn.codebook,
                )?;

                sh[3 + c] = rgb_coeff[0];
                sh[18 + c] = rgb_coeff[1];
                sh[33 + c] = rgb_coeff[2];
            }
        }

        gaussians.push(Gaussian3d {
            position,
            opacity,
            scale,
            _pad0: 0,
            rotation,
            sh,
        });
    }

    let mut min_scale = [f32::INFINITY; 3];
    let mut max_scale = [f32::NEG_INFINITY; 3];

    for g in &gaussians {
        for k in 0..3 {
            min_scale[k] = min_scale[k].min(g.scale[k]);
            max_scale[k] = max_scale[k].max(g.scale[k]);
        }
    }

    println!(
        "decoded internal scale_log min={:?}, max={:?}",
        min_scale, max_scale
    );

    Ok(Gaussians::Gaussian3d(gaussians))
}

fn ensure_supported_sog_format(meta: &SogMeta) -> anyhow::Result<()> {
    ensure!(
        meta.version == 2,
        "unsupported SOG version: {}",
        meta.version
    );

    ensure!(
        meta.scales.codebook.len() == 256,
        "invalid scales codebook length: {}",
        meta.scales.codebook.len()
    );

    ensure!(
        meta.sh0.codebook.len() == 256,
        "invalid sh0 codebook length: {}",
        meta.sh0.codebook.len()
    );

    if let Some(shn) = &meta.shn {
        ensure!(
            (1..=3).contains(&shn.bands),
            "invalid shN bands: {}",
            shn.bands
        );

        ensure!(shn.count <= 65536, "invalid shN count: {}", shn.count);

        ensure!(
            shn.codebook.len() == 256,
            "invalid shN codebook length: {}",
            shn.codebook.len()
        );
    }

    Ok(())
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn unlog(n: f32) -> f32 {
    n.signum() * (n.abs().exp() - 1.0)
}

fn decode_position(
    means_l_rgb: [u8; 3],
    means_u_rgb: [u8; 3],
    mins: [f32; 3],
    maxs: [f32; 3],
) -> [f32; 3] {
    let qx = ((means_u_rgb[0] as u16) << 8) | means_l_rgb[0] as u16;
    let qy = ((means_u_rgb[1] as u16) << 8) | means_l_rgb[1] as u16;
    let qz = ((means_u_rgb[2] as u16) << 8) | means_l_rgb[2] as u16;

    let nx = lerp(mins[0], maxs[0], qx as f32 / 65535.0);
    let ny = lerp(mins[1], maxs[1], qy as f32 / 65535.0);
    let nz = lerp(mins[2], maxs[2], qz as f32 / 65535.0);

    [unlog(nx), unlog(ny), unlog(nz)]
}

fn decode_quat_component(c: u8) -> f32 {
    ((c as f32) / 255.0 - 0.5) * std::f32::consts::SQRT_2
}

fn decode_orientation(rgba: [u8; 4]) -> anyhow::Result<[f32; 4]> {
    let a = decode_quat_component(rgba[0]);
    let b = decode_quat_component(rgba[1]);
    let c = decode_quat_component(rgba[2]);

    ensure!(
        rgba[3] >= 252,
        "invalid SOG quaternion mode byte: {}",
        rgba[3]
    );

    let mode = rgba[3] - 252;

    ensure!(mode <= 3, "invalid SOG quaternion mode: {}", mode);

    let t = a * a + b * b + c * c;
    let d = (1.0 - t).max(0.0).sqrt();

    let q = match mode {
        0 => [d, a, b, c], // omitted = x
        1 => [a, d, b, c], // omitted = y
        2 => [a, b, d, c], // omitted = z
        3 => [a, b, c, d], // omitted = w
        _ => unreachable!(),
    };

    Ok(q)
}

fn decode_shn_label(label_rgb: [u8; 3]) -> usize {
    label_rgb[0] as usize + ((label_rgb[1] as usize) << 8)
}

fn decode_shn_coeff_rgb(
    label: usize,
    coeff_index: usize,
    coeff_count: usize,
    centroids: &ImageData,
    codebook: &[f32],
) -> anyhow::Result<[f32; 3]> {
    let u = (label % 64) * coeff_count + coeff_index;
    let v = label / 64;

    ensure!(
        u < centroids.width as usize && v < centroids.height as usize,
        "shN centroid pixel out of bounds: label={}, coeff={}, u={}, v={}, image={}x{}",
        label,
        coeff_index,
        u,
        v,
        centroids.width,
        centroids.height
    );

    let pixel_index = v * centroids.width as usize + u;
    let rgb = centroids.pixel_rgb(pixel_index)?;

    Ok([
        codebook[rgb[0] as usize],
        codebook[rgb[1] as usize],
        codebook[rgb[2] as usize],
    ])
}

fn alpha_to_raw_opacity(alpha: f32) -> f32 {
    let a = alpha.clamp(1e-6, 1.0 - 1e-6);
    (a / (1.0 - a)).ln()
}

fn shn_coeff_count(bands: u32) -> anyhow::Result<usize> {
    match bands {
        1 => Ok(3),
        2 => Ok(8),
        3 => Ok(15),
        _ => bail!("invalid shN bands: {bands}"),
    }
}
