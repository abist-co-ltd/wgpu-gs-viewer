use crate::gaussian;
use anyhow::{Context, Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyPayloadKind {
    Gaussian3d,
    Gaussian4d,
    Unknown,
}

#[derive(Debug)]
struct PlyHeader {
    format: PlyFormat,
    header_end: usize,
    vertex_layout: VertexLayout,
    payload_kind: PlyPayloadKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyScalarType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyScalarType {
    fn size(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PropRead {
    offset: usize,
    ty: PlyScalarType,
}

#[derive(Debug)]
struct VertexLayout {
    count: usize,
    stride: usize,
    props: Vec<VertexProp>,
}

impl VertexLayout {
    fn find(&self, name: &str) -> Option<PropRead> {
        self.props
            .iter()
            .find(|p| p.name == name)
            .map(|p| PropRead {
                offset: p.offset,
                ty: p.ty,
            })
    }

    fn required(&self, name: &str) -> Result<PropRead> {
        self.find(name)
            .with_context(|| format!("missing required PLY vertex property '{name}'"))
    }

    fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }
}

#[derive(Debug)]
struct VertexProp {
    name: String,
    ty: PlyScalarType,
    offset: usize,
}

#[derive(Debug)]
struct Gaussian3dLayout {
    vertex_count: usize,
    vertex_stride: usize,

    x: PropRead,
    y: PropRead,
    z: PropRead,

    opacity: PropRead,

    scale_0: PropRead,
    scale_1: PropRead,
    scale_2: PropRead,

    rot_0: PropRead,
    rot_1: PropRead,
    rot_2: PropRead,
    rot_3: PropRead,

    f_dc: [Option<PropRead>; 3],
    f_rest: [Option<PropRead>; 45],
}

impl Gaussian3dLayout {
    fn from_vertex_layout(layout: &VertexLayout) -> Result<Self> {
        let mut f_dc = [None; 3];
        for (i, slot) in f_dc.iter_mut().enumerate() {
            let name = format!("f_dc_{i}");
            *slot = layout.find(&name);
        }

        let mut f_rest = [None; 45];
        for (i, slot) in f_rest.iter_mut().enumerate() {
            let name = format!("f_rest_{i}");
            *slot = layout.find(&name);
        }

        Ok(Self {
            vertex_count: layout.count,
            vertex_stride: layout.stride,

            x: layout.required("x")?,
            y: layout.required("y")?,
            z: layout.required("z")?,

            opacity: layout.required("opacity")?,

            scale_0: layout.required("scale_0")?,
            scale_1: layout.required("scale_1")?,
            scale_2: layout.required("scale_2")?,

            rot_0: layout.required("rot_0")?,
            rot_1: layout.required("rot_1")?,
            rot_2: layout.required("rot_2")?,
            rot_3: layout.required("rot_3")?,

            f_dc,
            f_rest,
        })
    }
}

#[cfg(target_arch = "wasm32")]
fn format_url(filename: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let href = location.href().unwrap();
    let base = reqwest::Url::parse(&href).unwrap();

    base.join("assets/")
        .unwrap()
        .join(filename.trim_start_matches('/'))
        .unwrap()
}

pub fn parse_3dgs_ply_bytes(data: &[u8]) -> anyhow::Result<Vec<gaussian::Gaussian3d>> {
    let header = parse_ply_header(data)?;

    match header.format {
        PlyFormat::BinaryLittleEndian => {}
        PlyFormat::Ascii => {
            bail!("ASCII PLY is not supported by this loader");
        }
        PlyFormat::BinaryBigEndian => {
            bail!("binary_big_endian PLY is not supported by this loader");
        }
    }

    match header.payload_kind {
        PlyPayloadKind::Gaussian3d => parse_3dgs_binary_le(data, &header),
        PlyPayloadKind::Gaussian4d => {
            bail!("4DGS PLY was detected, but 4DGS loader is not implemented yet");
        }
        PlyPayloadKind::Unknown => {
            bail!("unsupported PLY payload type");
        }
    }
}

fn parse_3dgs_binary_le(data: &[u8], header: &PlyHeader) -> Result<Vec<gaussian::Gaussian3d>> {
    let layout = Gaussian3dLayout::from_vertex_layout(&header.vertex_layout)?;

    let body_offset = header.header_end;

    let expected_body_size = layout
        .vertex_count
        .checked_mul(layout.vertex_stride)
        .context("PLY vertex body size overflow")?;

    let expected_total_size = body_offset
        .checked_add(expected_body_size)
        .context("PLY total size overflow")?;

    if data.len() < expected_total_size {
        bail!(
            "PLY body is too short: expected at least {}, got {}",
            expected_total_size,
            data.len()
        );
    }

    let mut gaussians = Vec::with_capacity(layout.vertex_count);

    for i in 0..layout.vertex_count {
        let base = body_offset + i * layout.vertex_stride;

        let read = |p: PropRead| -> f32 { read_scalar_as_f32(data, base + p.offset, p.ty) };

        let read_opt = |p: Option<PropRead>| -> f32 { p.map(read).unwrap_or(0.0) };

        let mut sh = [0.0; gaussian::SH_COUNT];

        sh[0] = read_opt(layout.f_dc[0]);
        sh[1] = read_opt(layout.f_dc[1]);
        sh[2] = read_opt(layout.f_dc[2]);

        for j in 0..45 {
            sh[3 + j] = read_opt(layout.f_rest[j]);
        }

        gaussians.push(gaussian::Gaussian3d {
            position: [read(layout.x), read(layout.y), read(layout.z)],
            opacity: read(layout.opacity),

            scale: [
                read(layout.scale_0),
                read(layout.scale_1),
                read(layout.scale_2),
            ],
            _pad0: 0,

            rotation: [
                read(layout.rot_0),
                read(layout.rot_1),
                read(layout.rot_2),
                read(layout.rot_3),
            ],

            sh,
        });
    }

    Ok(gaussians)
}

fn parse_ply_header(data: &[u8]) -> Result<PlyHeader> {
    let header_end = find_header_end(data).context("PLY header does not contain end_header")?;

    let header_bytes = &data[..header_end];
    let header_text = std::str::from_utf8(header_bytes).context("PLY header is not valid UTF-8")?;

    let mut format: Option<PlyFormat> = None;

    let mut in_vertex = false;
    let mut vertex_count: Option<usize> = None;
    let mut vertex_props: Vec<VertexProp> = Vec::new();
    let mut vertex_stride = 0usize;

    for raw_line in header_text.lines() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with("comment ") || line.starts_with("obj_info ") {
            continue;
        }

        if line == "ply" {
            continue;
        }

        if line.starts_with("format ") {
            format = Some(parse_ply_format(line)?);
            continue;
        }

        if line.starts_with("element ") {
            let mut it = line.split_whitespace();

            let _element = it.next();
            let name = it.next();
            let count = it.next();

            if name == Some("vertex") {
                let count = count
                    .context("vertex element does not have count")?
                    .parse::<usize>()
                    .context("failed to parse vertex count")?;

                in_vertex = true;
                vertex_count = Some(count);
                vertex_props.clear();
                vertex_stride = 0;
            } else {
                in_vertex = false;
            }

            continue;
        }

        if in_vertex && line.starts_with("property ") {
            let tokens: Vec<_> = line.split_whitespace().collect();

            if tokens.len() >= 2 && tokens[1] == "list" {
                bail!("list property in vertex is not supported");
            }

            if tokens.len() != 3 {
                bail!("unsupported vertex property line: {line}");
            }

            let ty = parse_scalar_type(tokens[1])
                .with_context(|| format!("unsupported PLY scalar type: {}", tokens[1]))?;

            let name = tokens[2].to_string();
            let offset = vertex_stride;

            vertex_props.push(VertexProp { name, ty, offset });

            vertex_stride += ty.size();

            continue;
        }
    }

    let format = format.context("PLY header does not contain format line")?;

    let vertex_layout = VertexLayout {
        count: vertex_count.context("PLY does not contain vertex element")?,
        stride: vertex_stride,
        props: vertex_props,
    };

    let payload_kind = classify_ply_payload(&vertex_layout);

    Ok(PlyHeader {
        format,
        header_end,
        vertex_layout,
        payload_kind,
    })
}

fn parse_ply_format(line: &str) -> Result<PlyFormat> {
    let mut it = line.split_whitespace();

    let keyword = it.next();
    let format = it.next();
    let version = it.next();

    if keyword != Some("format") {
        bail!("invalid PLY format line: {line}");
    }

    if version != Some("1.0") {
        bail!("unsupported PLY format version in line: {line}");
    }

    match format {
        Some("ascii") => Ok(PlyFormat::Ascii),
        Some("binary_little_endian") => Ok(PlyFormat::BinaryLittleEndian),
        Some("binary_big_endian") => Ok(PlyFormat::BinaryBigEndian),
        _ => bail!("unsupported PLY format line: {line}"),
    }
}

fn classify_ply_payload(layout: &VertexLayout) -> PlyPayloadKind {
    let has_3dgs_core = layout.has("x")
        && layout.has("y")
        && layout.has("z")
        && layout.has("opacity")
        && layout.has("scale_0")
        && layout.has("scale_1")
        && layout.has("scale_2")
        && layout.has("rot_0")
        && layout.has("rot_1")
        && layout.has("rot_2")
        && layout.has("rot_3");

    if !has_3dgs_core {
        return PlyPayloadKind::Unknown;
    }

    // TODO:
    let has_4dgs_markers = layout.has("trbf_center")
        || layout.has("trbf_scale")
        || layout.has("motion_0")
        || layout.has("motion_1")
        || layout.has("motion_2")
        || layout.has("motion_3")
        || layout.has("motion_4")
        || layout.has("motion_5")
        || layout.has("motion_6")
        || layout.has("motion_7")
        || layout.has("motion_8")
        || layout.has("omega_0")
        || layout.has("omega_1")
        || layout.has("omega_2")
        || layout.has("omega_3");

    if has_4dgs_markers {
        PlyPayloadKind::Gaussian4d
    } else {
        PlyPayloadKind::Gaussian3d
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    const LF: &[u8] = b"end_header\n";
    const CRLF: &[u8] = b"end_header\r\n";

    if let Some(pos) = find_bytes(data, CRLF) {
        return Some(pos + CRLF.len());
    }

    if let Some(pos) = find_bytes(data, LF) {
        return Some(pos + LF.len());
    }

    None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_scalar_type(s: &str) -> Result<PlyScalarType> {
    let ty = match s {
        "char" | "int8" => PlyScalarType::Char,
        "uchar" | "uint8" => PlyScalarType::UChar,
        "short" | "int16" => PlyScalarType::Short,
        "ushort" | "uint16" => PlyScalarType::UShort,
        "int" | "int32" => PlyScalarType::Int,
        "uint" | "uint32" => PlyScalarType::UInt,
        "float" | "float32" => PlyScalarType::Float,
        "double" | "float64" => PlyScalarType::Double,
        _ => bail!("unknown scalar type: {s}"),
    };

    Ok(ty)
}

fn read_scalar_as_f32(data: &[u8], offset: usize, ty: PlyScalarType) -> f32 {
    match ty {
        PlyScalarType::Char => i8::from_le_bytes([data[offset]]) as f32,
        PlyScalarType::UChar => data[offset] as f32,
        PlyScalarType::Short => i16::from_le_bytes([data[offset], data[offset + 1]]) as f32,
        PlyScalarType::UShort => u16::from_le_bytes([data[offset], data[offset + 1]]) as f32,
        PlyScalarType::Int => i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as f32,
        PlyScalarType::UInt => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as f32,
        PlyScalarType::Float => f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
        PlyScalarType::Double => f64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as f32,
    }
}
