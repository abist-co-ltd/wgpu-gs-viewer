use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::TileGpuResources;
use crate::renderer::tile::resources::tile_pipeline_resources::TileRangeParams;
pub struct TileRangeBindings {
    pub clear_tile_ranges_bind_group_layout: wgpu::BindGroupLayout,
    pub clear_tile_ranges_bind_group: wgpu::BindGroup,

    pub tile_range_bind_group_layout: wgpu::BindGroupLayout,
    pub tile_range_bind_group: wgpu::BindGroup,

    pub tile_range_params_buffer: wgpu::Buffer,
}

impl TileRangeBindings {
    pub fn new(gpu: &GpuContext, resources: &TileGpuResources) -> Self {
        let tile_range_params_buffer = gpu.create_buffer_init(
            "Tile Range Params Buffer",
            bytemuck::bytes_of(&TileRangeParams {
                tile_count: resources.pipeline.tile_count,
            }),
            wgpu::BufferUsages::UNIFORM,
        );

        let clear_tile_ranges_bind_group_layout = gpu.create_bind_group_layout(
            "clear tile ranges bind group layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let clear_tile_ranges_bind_group = Self::make_clear_tile_ranges_bind_group(
            gpu,
            &clear_tile_ranges_bind_group_layout,
            &tile_range_params_buffer,
            resources,
        );

        let tile_range_bind_group_layout = gpu.create_bind_group_layout(
            "tile range bind group layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let tile_range_bind_group = Self::make_tile_range_bind_group(
            gpu,
            &tile_range_bind_group_layout,
            &tile_range_params_buffer,
            resources,
        );

        Self {
            clear_tile_ranges_bind_group_layout,
            clear_tile_ranges_bind_group,
            tile_range_bind_group_layout,
            tile_range_bind_group,
            tile_range_params_buffer,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &TileGpuResources) {
        self.tile_range_params_buffer = gpu.create_buffer_init(
            "Tile Range Params Buffer",
            bytemuck::bytes_of(&TileRangeParams {
                tile_count: resources.pipeline.tile_count,
            }),
            wgpu::BufferUsages::UNIFORM,
        );

        self.clear_tile_ranges_bind_group = Self::make_clear_tile_ranges_bind_group(
            gpu,
            &self.clear_tile_ranges_bind_group_layout,
            &self.tile_range_params_buffer,
            resources,
        );
        self.tile_range_bind_group = Self::make_tile_range_bind_group(
            gpu,
            &self.tile_range_bind_group_layout,
            &self.tile_range_params_buffer,
            resources,
        );
    }

    fn make_clear_tile_ranges_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        tile_range_params_buffer: &wgpu::Buffer,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "clear tile ranges bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_range_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.tile_ranges_buffer.as_entire_binding(),
                },
            ],
        )
    }

    fn make_tile_range_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        tile_range_params_buffer: &wgpu::Buffer,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "tile range bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.pipeline.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_range_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.pair_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.tile_ranges_buffer.as_entire_binding(),
                },
            ],
        )
    }
}
