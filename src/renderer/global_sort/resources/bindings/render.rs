use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::GlobalSortGpuResources;

pub struct RenderBindings {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl RenderBindings {
    pub fn new(gpu: &GpuContext, resources: &GlobalSortGpuResources) -> Self {
        let bind_group_layout = gpu.create_bind_group_layout(
            "Global Sort Render Bind Group Layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let bind_group = gpu.create_bind_group(
            "Global Sort Render Bind Group",
            &bind_group_layout,
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
                    resource: resources.pipeline.sort_values_buffer.as_entire_binding(),
                },
            ],
        );

        Self {
            bind_group_layout,
            bind_group,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &GlobalSortGpuResources) {
        self.bind_group = Self::make_bind_group(gpu, &self.bind_group_layout, resources);
    }

    fn make_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        resources: &GlobalSortGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "Global Sort Render Bind Group",
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
                    resource: resources.pipeline.sort_values_buffer.as_entire_binding(),
                },
            ],
        )
    }
}
