use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::TileGpuResources;
use crate::renderer::tile::resources::tile_pipeline_resources::DuplicateParams;

pub struct DuplicateBindings {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    duplicate_params_buffer: wgpu::Buffer,
}

impl DuplicateBindings {
    pub fn new(gpu: &GpuContext, resources: &TileGpuResources) -> Self {
        let duplicate_params_buffer = gpu.create_buffer_init(
            "Duplicate Params Buffer",
            bytemuck::bytes_of(&DuplicateParams {
                tiles_width: resources.pipeline.tiles_width,
                tiles_height: resources.pipeline.tiles_height,
                max_pairs: resources.pipeline.max_pairs,
                _pad: 0,
            }),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group_layout = gpu.create_bind_group_layout(
            "duplicate bind group layout",
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let bind_group =
            Self::make_bind_group(gpu, &bind_group_layout, &duplicate_params_buffer, resources);

        Self {
            bind_group_layout,
            bind_group,
            duplicate_params_buffer,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &TileGpuResources) {
        self.duplicate_params_buffer = gpu.create_buffer_init(
            "Duplicate Params Buffer",
            bytemuck::bytes_of(&DuplicateParams {
                tiles_width: resources.pipeline.tiles_width,
                tiles_height: resources.pipeline.tiles_height,
                max_pairs: resources.pipeline.max_pairs,
                _pad: 0,
            }),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        self.bind_group = Self::make_bind_group(
            gpu,
            &self.bind_group_layout,
            &self.duplicate_params_buffer,
            resources,
        );
    }

    fn make_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        duplicate_params_buffer: &wgpu::Buffer,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "duplicate bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: duplicate_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources
                        .pipeline
                        .preprocess_output_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.pair_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.pipeline.pair_values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.pipeline.visible_count_buffer.as_entire_binding(),
                },
            ],
        )
    }
}
