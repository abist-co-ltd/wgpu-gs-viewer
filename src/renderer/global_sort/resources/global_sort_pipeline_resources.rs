use super::RADIX_SORT_PASSES;
use crate::gpu::context::GpuContext;

pub struct GlobalSortPipelineResources {
    pub preprocess_output_buffer: wgpu::Buffer,

    pub sort_keys_buffer: wgpu::Buffer,
    pub sort_values_buffer: wgpu::Buffer,

    pub sort_keys_tmp_buffer: wgpu::Buffer,
    pub sort_values_tmp_buffer: wgpu::Buffer,

    pub visible_count_buffer: wgpu::Buffer,

    pub radix_params_buffer: wgpu::Buffer,
    pub radix_dispatch_args_buffer: wgpu::Buffer,
    pub draw_indirect_args_buffer: wgpu::Buffer,

    pub gaussian_count: u32,
}

impl GlobalSortPipelineResources {
    pub fn new(gpu: &GpuContext, gaussian_count: u32) -> Self {
        let capacity = gaussian_count.max(1);

        let preprocess_output_buffer = gpu.create_buffer(
            "Global Sort Preprocess Output Buffer",
            std::mem::size_of::<PreprocessOutput>() as u64 * capacity as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let sort_keys_buffer = gpu.create_buffer(
            "Global Sort Sort Keys Buffer",
            std::mem::size_of::<u32>() as u64 * capacity as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let sort_values_buffer = gpu.create_buffer(
            "Global Sort Sort Values Buffer",
            std::mem::size_of::<u32>() as u64 * capacity as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let sort_keys_tmp_buffer = gpu.create_buffer(
            "Global Sort Sort Keys Tmp Buffer",
            std::mem::size_of::<u32>() as u64 * capacity as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let sort_values_tmp_buffer = gpu.create_buffer(
            "Global Sort Sort Values Tmp Buffer",
            std::mem::size_of::<u32>() as u64 * capacity as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let visible_count_buffer = gpu.create_buffer(
            "Global Sort Visible Count Buffer",
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let radix_dispatch_args_buffer = gpu.create_buffer(
            "Global Sort Radix Dispatch Args Buffer",
            std::mem::size_of::<wgpu::util::DispatchIndirectArgs>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );

        let radix_params_buffer = gpu.create_buffer(
            "Global Sort Radix Params Buffer",
            std::mem::size_of::<RadixSortParams>() as u64 * RADIX_SORT_PASSES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let draw_indirect_args_buffer = gpu.create_buffer(
            "Global Sort Draw Indirect Args Buffer",
            std::mem::size_of::<DrawIndirectArgs>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            preprocess_output_buffer,

            sort_keys_buffer,
            sort_values_buffer,

            sort_keys_tmp_buffer,
            sort_values_tmp_buffer,

            visible_count_buffer,

            radix_params_buffer,
            radix_dispatch_args_buffer,
            draw_indirect_args_buffer,

            gaussian_count,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PreprocessOutput {
    pub center_depth: [f32; 4],
    pub elipse_axis: [f32; 4],
    pub color_opacity: [f32; 4],
}

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
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
