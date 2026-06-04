use crate::scene;

pub const SH_COUNT: usize = 48;

pub const PREFIX_DISPATCH_ARGS_COUNT: u64 = 5;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PreprocessOutput {
    pub conic_opacity: [f32; 4],
    pub color_radius: [f32; 4],
    pub tile_rect: [u32; 4],
    pub uv_depth: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrefixCounts {
    pub n0: u32,
    pub n1: u32,
    pub n2: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrefixLevelParams {
    pub level: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PairKey {
    pub tile_id: u32,
    pub depth_bits: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DuplicateParams {
    pub tiles_width: u32,
    pub tiles_height: u32,
    pub max_pairs: u32,
    pub _pad: u32,
}

pub const RADIX_SORT_BINS: u32 = 256;
pub const RADIX_SORT_PASSES: usize = 8;
pub const MAX_RADIX_WORKGROUPS: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixSortParams {
    pub num_elements: u32,
    pub shift: u32,
    pub num_workgroups: u32,
    pub num_blocks_per_workgroup: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixPassIndex {
    pub value: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TotalPairs {
    pub raw_total_pairs: u32,
    pub sort_pair_count: u32,
    pub visible_count: u32,
    pub overflow: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PairCountParams {
    pub max_pairs: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileRange {
    pub start: u32,
    pub end: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileRangeParams {
    pub tile_count: u32,
}

pub struct GaussianResources {
    pub gaussian_buffer: wgpu::Buffer,

    pub preprocess_output_buffer: wgpu::Buffer,
    pub tiles_touched_buffer: wgpu::Buffer,
    pub visible_count_buffer: wgpu::Buffer,

    pub prefix_dispatch_args_buffer: wgpu::Buffer,
    pub prefix_counts_buffer: wgpu::Buffer,

    pub offsets_buffer: wgpu::Buffer,
    pub block_sums0_buffer: wgpu::Buffer,
    pub block_offsets0_buffer: wgpu::Buffer,
    pub block_sums1_buffer: wgpu::Buffer,
    pub block_offsets1_buffer: wgpu::Buffer,
    pub block_sums2_buffer: wgpu::Buffer,

    pub pair_keys_buffer: wgpu::Buffer,
    pub pair_values_buffer: wgpu::Buffer,
    pub pair_keys_tmp_buffer: wgpu::Buffer,
    pub pair_values_tmp_buffer: wgpu::Buffer,

    pub radix_histograms_buffer: wgpu::Buffer,
    pub total_pairs_buffer: wgpu::Buffer,
    pub radix_params_buffer: wgpu::Buffer,
    pub radix_dispatch_args_buffer: wgpu::Buffer,
    pub tile_range_dispatch_args_buffer: wgpu::Buffer,

    pub tile_ranges_buffer: wgpu::Buffer,

    pub gaussian_count: u32,
    pub max_pairs: u32,
    pub tile_count: u32,
    pub tiles_width: u32,
    pub tiles_height: u32,
}

impl GaussianResources {
    pub fn new(device: &wgpu::Device, gaussians: &Gaussians) -> Self {
        use wgpu::util::DeviceExt;

        let gaussian_count = gaussians.len() as u32;
        let count1 = gaussian_count.max(1);

        // TODO:
        //  max_pairs is fixed for now.
        //  If overflow happens, consider async readback of raw_total_pairs and resizing
        //  pair/sort buffers for future frames. Avoid waiting for readback or resizing
        //  synchronously every frame.
        let max_pairs = count1 * 32;

        let max_blocks0 = count1.div_ceil(256);
        let max_blocks1 = max_blocks0.div_ceil(256);
        let max_blocks2 = max_blocks1.div_ceil(256);

        let radix_num_workgroups = MAX_RADIX_WORKGROUPS;
        let radix_histogram_len = radix_num_workgroups * RADIX_SORT_BINS;

        let tiles_width = scene::SCREEN_WIDTH.div_ceil(16);
        let tiles_height = scene::SCREEN_HEIGHT.div_ceil(16);
        let tile_count = tiles_width * tiles_height;

        let gaussian_buffer = if gaussians.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gaussian Buffer"),
                size: gaussians.size_of_type(),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gaussian Buffer"),
                contents: gaussians.cast_slice(),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let preprocess_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Preprocess Output Buffer"),
            size: std::mem::size_of::<PreprocessOutput>() as u64 * count1 as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tiles_touched_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tiles Touched Buffer"),
            size: std::mem::size_of::<u32>() as u64 * count1 as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Count Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefix_dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefix Dispatch Args Buffer"),
            size: std::mem::size_of::<DispatchIndirectArgs>() as u64 * PREFIX_DISPATCH_ARGS_COUNT,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prefix_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefix Counts Buffer"),
            size: std::mem::size_of::<PrefixCounts>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefix Offsets Buffer"),
            size: std::mem::size_of::<u32>() as u64 * count1 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Sums 0 Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_blocks0 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_offsets0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Offsets 0 Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_blocks0 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Sums 1 Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_blocks1 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_offsets1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Offsets 1 Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_blocks1 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums2_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Sums 2 Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_blocks2 as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pair_keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pair Keys Buffer"),
            size: std::mem::size_of::<PairKey>() as u64 * max_pairs as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pair_values_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pair Values Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_pairs as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pair_keys_tmp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pair Keys Tmp Buffer"),
            size: std::mem::size_of::<PairKey>() as u64 * max_pairs as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pair_values_tmp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pair Values Tmp Buffer"),
            size: std::mem::size_of::<u32>() as u64 * max_pairs as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radix_histograms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Histograms Buffer"),
            size: std::mem::size_of::<u32>() as u64 * radix_histogram_len as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Total Pairs Buffer"),
            size: std::mem::size_of::<TotalPairs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radix_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Params Buffer"),
            size: std::mem::size_of::<RadixSortParams>() as u64 * RADIX_SORT_PASSES as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radix_dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Dispatch Args Buffer"),
            size: std::mem::size_of::<DispatchIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_range_dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Range Dispatch Args Buffer"),
            size: std::mem::size_of::<DispatchIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_ranges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Ranges Buffer"),
            size: std::mem::size_of::<TileRange>() as u64 * tile_count as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            gaussian_buffer,
            preprocess_output_buffer,
            tiles_touched_buffer,
            visible_count_buffer,
            prefix_dispatch_args_buffer,
            prefix_counts_buffer,
            offsets_buffer,
            block_sums0_buffer,
            block_offsets0_buffer,
            block_sums1_buffer,
            block_offsets1_buffer,
            block_sums2_buffer,
            pair_keys_buffer,
            pair_values_buffer,
            pair_keys_tmp_buffer,
            pair_values_tmp_buffer,
            radix_histograms_buffer,
            total_pairs_buffer,
            radix_params_buffer,
            radix_dispatch_args_buffer,
            tile_range_dispatch_args_buffer,
            tile_ranges_buffer,
            gaussian_count,
            max_pairs,
            tile_count,
            tiles_width,
            tiles_height,
        }
    }
}
