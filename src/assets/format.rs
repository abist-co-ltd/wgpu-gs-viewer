#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Ply,
    Sog,
    // TODO:
    // Spz
}

impl FileFormat {
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }

    pub fn from_file_name(file_name: &str) -> Option<Self> {
        let ext = file_name.rsplit_once('.')?.1;
        Self::from_extension(ext)
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "ply" => Some(Self::Ply),
            "sog" => Some(Self::Sog),
            _ => None,
        }
    }
}
