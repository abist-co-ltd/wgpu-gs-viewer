use std::path::Path;

use anyhow::Context;

use super::format::FileFormat;
use super::ply_loader;
use crate::assets::sog_loader;
use crate::gaussian;

pub fn load_gaussians_from_path(
    format: FileFormat,
    path: &Path,
) -> anyhow::Result<gaussian::Gaussians> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read file: {path:?}"))?;
    load_gaussians_from_bytes(format, &bytes)
}

pub fn load_gaussians_from_bytes(
    format: FileFormat,
    bytes: &[u8],
) -> anyhow::Result<gaussian::Gaussians> {
    match format {
        FileFormat::Ply => ply_loader::parse_gaussian_ply_bytes(bytes)
            .with_context(|| "failed to parse PLY gaussian file"),
        FileFormat::Sog => sog_loader::load_sog_from_bytes(bytes)
            .with_context(|| "failed to load SOG gaussian file"),
    }
}
