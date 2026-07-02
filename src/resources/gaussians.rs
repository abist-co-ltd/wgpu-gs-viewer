use crate::gpu::context::GpuContext;

pub struct GaussianResource {
    pub gaussian_buffer: wgpu::Buffer,
}

impl GaussianResource {
    pub fn new(gpu: &GpuContext, gaussians: &Gaussians) -> Self {
        let gaussian_buffer = if gaussians.is_empty() {
            gpu.create_buffer(
                "Gaussian Scene Buffer",
                gaussians.size_of_type(),
                wgpu::BufferUsages::STORAGE,
            )
        } else {
            gpu.create_buffer_init(
                "Gaussian Scene Buffer",
                gaussians.cast_slice(),
                wgpu::BufferUsages::STORAGE,
            )
        };

        Self { gaussian_buffer }
    }
}

pub const SH_COUNT: usize = 48;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gaussian3d {
    pub position: [f32; 3],
    pub opacity: f32,

    pub scale: [f32; 3],
    pub _pad0: u32,

    pub rotation: [f32; 4],
    pub sh: [f32; SH_COUNT],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gaussian4d {
    pub position: [f32; 3],
    pub opacity: f32,

    pub scale: [f32; 3],
    pub _pad0: u32,

    pub rotation: [f32; 4],

    pub motion_0: [f32; 3], // motion_0, motion_1, motion_2
    pub _pad1: u32,
    pub motion_1: [f32; 3], // motion_3, motion_4, motion_5
    pub _pad2: u32,
    pub motion_2: [f32; 3], // motion_6, motion_7, motion_8
    pub _pad3: u32,

    pub omega: [f32; 4],
    pub trbf_center: f32,
    pub trbf_scale: f32,
    pub _pad4: u32,
    pub _pad5: u32,

    pub base_color: [f32; 3], // f_dc_0, f_dc_1, f_dc_2
    pub _pad6: u32,
}

pub enum Gaussians {
    Gaussian3d(Vec<Gaussian3d>),
    Gaussian4d(Vec<Gaussian4d>),
}

impl Gaussians {
    pub fn len(&self) -> usize {
        match self {
            Gaussians::Gaussian3d(a) => a.len(),
            Gaussians::Gaussian4d(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cast_slice(&self) -> &[u8] {
        match self {
            Gaussians::Gaussian3d(a) => bytemuck::cast_slice(a),
            Gaussians::Gaussian4d(a) => bytemuck::cast_slice(a),
        }
    }

    pub fn size_of_type(&self) -> u64 {
        match &self {
            Gaussians::Gaussian3d(_) => std::mem::size_of::<Gaussian3d>() as u64,
            Gaussians::Gaussian4d(_) => std::mem::size_of::<Gaussian4d>() as u64,
        }
    }
}
