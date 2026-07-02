use crate::gpu::{context::GpuContext, frame_targets::FrameTargets};
use crate::renderer::tile::resources::TileGpuResources;

pub struct TileRenderBindings {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl TileRenderBindings {
    pub fn new(
        gpu: &GpuContext,
        frame_target: &FrameTargets,
        resources: &TileGpuResources,
    ) -> Self {
        let bind_group_layout = gpu.create_bind_group_layout(
            "tile render bind group layout",
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let bind_group = Self::make_bind_group(
            gpu,
            &bind_group_layout,
            &frame_target.render_texture_view,
            resources,
        );

        Self {
            bind_group_layout,
            bind_group,
        }
    }

    pub fn recreate(
        &mut self,
        gpu: &GpuContext,
        frame_target: &FrameTargets,
        resources: &TileGpuResources,
    ) {
        self.bind_group = Self::make_bind_group(
            gpu,
            &self.bind_group_layout,
            &frame_target.render_texture_view,
            resources,
        );
    }

    fn make_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        render_texture_view: &wgpu::TextureView,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "tile render bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.scene.uniform_buffer.as_entire_binding(),
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
                    resource: resources.pipeline.tile_ranges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.pair_values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.pipeline.total_pairs_buffer.as_entire_binding(),
                },
            ],
        )
    }
}
