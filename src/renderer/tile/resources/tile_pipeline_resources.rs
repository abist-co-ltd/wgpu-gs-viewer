use crate::gpu::context::GpuContext;

use super::{PREFIX_DISPATCH_ARGS_COUNT, RADIX_SORT_PASSES};

pub struct TilePipelineResources {
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

impl TilePipelineResources {
    pub fn new(gpu: &GpuContext, gaussian_count: u32) -> Self {
        let gaussian_count = gaussian_count.max(1);

        // TODO:
        //  max_pairs is fixed for now.
        //  If overflow happens, consider async readback of raw_total_pairs and resizing
        //  pair/sort buffers for future frames. Avoid waiting for readback or resizing
        //  synchronously every frame.
        let max_pairs = gaussian_count * 32;

        let max_blocks0 = gaussian_count.div_ceil(256);
        let max_blocks1 = max_blocks0.div_ceil(256);
        let max_blocks2 = max_blocks1.div_ceil(256);

        let radix_num_workgroups = MAX_RADIX_WORKGROUPS;
        let radix_histogram_len = radix_num_workgroups * RADIX_SORT_BINS;

        let surface_size = gpu.surface_size();

        let width = surface_size.width.max(1);
        let height = surface_size.height.max(1);

        let tiles_width = width.div_ceil(16);
        let tiles_height = height.div_ceil(16);
        let tile_count = tiles_width * tiles_height;

        let preprocess_output_buffer = gpu.create_buffer(
            "Preprocess Output Buffer",
            std::mem::size_of::<PreprocessOutput>() as u64 * gaussian_count as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let tiles_touched_buffer = gpu.create_buffer(
            "Tiles Touched Buffer",
            std::mem::size_of::<u32>() as u64 * gaussian_count as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let visible_count_buffer = gpu.create_buffer(
            "Visible Count Buffer",
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let prefix_dispatch_args_buffer = gpu.create_buffer(
            "Prefix Dispatch Args Buffer",
            std::mem::size_of::<DispatchIndirectArgs>() as u64 * PREFIX_DISPATCH_ARGS_COUNT,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let prefix_counts_buffer = gpu.create_buffer(
            "Prefix Counts Buffer",
            std::mem::size_of::<PrefixCounts>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let offsets_buffer = gpu.create_buffer(
            "Prefix Offsets Buffer",
            std::mem::size_of::<u32>() as u64 * gaussian_count as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let block_sums0_buffer = gpu.create_buffer(
            "Block Sums 0 Buffer",
            std::mem::size_of::<u32>() as u64 * max_blocks0 as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let block_offsets0_buffer = gpu.create_buffer(
            "Block Offsets 0 Buffer",
            std::mem::size_of::<u32>() as u64 * max_blocks0 as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let block_sums1_buffer = gpu.create_buffer(
            "Block Sums 1 Buffer",
            std::mem::size_of::<u32>() as u64 * max_blocks1 as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let block_offsets1_buffer = gpu.create_buffer(
            "Block Offsets 1 Buffer",
            std::mem::size_of::<u32>() as u64 * max_blocks1 as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let block_sums2_buffer = gpu.create_buffer(
            "Block Sums 2 Buffer",
            std::mem::size_of::<u32>() as u64 * max_blocks2 as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let pair_keys_buffer = gpu.create_buffer(
            "Pair Keys Buffer",
            std::mem::size_of::<PairKey>() as u64 * max_pairs as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let pair_values_buffer = gpu.create_buffer(
            "Pair Values Buffer",
            std::mem::size_of::<u32>() as u64 * max_pairs as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let pair_keys_tmp_buffer = gpu.create_buffer(
            "Pair Keys Tmp Buffer",
            std::mem::size_of::<PairKey>() as u64 * max_pairs as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let pair_values_tmp_buffer = gpu.create_buffer(
            "Pair Values Tmp Buffer",
            std::mem::size_of::<u32>() as u64 * max_pairs as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let radix_histograms_buffer = gpu.create_buffer(
            "Radix Histograms Buffer",
            std::mem::size_of::<u32>() as u64 * radix_histogram_len as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let total_pairs_buffer = gpu.create_buffer(
            "Total Pairs Buffer",
            std::mem::size_of::<TotalPairs>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let radix_params_buffer = gpu.create_buffer(
            "Radix Params Buffer",
            std::mem::size_of::<RadixSortParams>() as u64 * RADIX_SORT_PASSES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let radix_dispatch_args_buffer = gpu.create_buffer(
            "Radix Dispatch Args Buffer",
            std::mem::size_of::<DispatchIndirectArgs>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let tile_range_dispatch_args_buffer = gpu.create_buffer(
            "Tile Range Dispatch Args Buffer",
            std::mem::size_of::<DispatchIndirectArgs>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let tile_ranges_buffer = gpu.create_buffer(
            "Tile Ranges Buffer",
            std::mem::size_of::<TileRange>() as u64 * tile_count as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Self {
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

    pub fn resize(&mut self, gpu: &GpuContext) {
        let surface_size = gpu.surface_size();

        let width = surface_size.width.max(1);
        let height = surface_size.height.max(1);

        self.tiles_width = width.div_ceil(16);
        self.tiles_height = height.div_ceil(16);
        self.tile_count = self.tiles_width * self.tiles_height;

        log::info!("tile pipeline resources resize size: ({width}, {height})");

        self.tile_ranges_buffer = gpu.create_buffer(
            "Tile Ranges Buffer",
            std::mem::size_of::<TileRange>() as u64 * self.tile_count as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
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

const RADIX_SORT_BINS: u32 = 256;
const MAX_RADIX_WORKGROUPS: u32 = 256;

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
