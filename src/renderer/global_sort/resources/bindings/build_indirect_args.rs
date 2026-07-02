use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::GlobalSortGpuResources;

pub struct BuildIndirectArgsBindings {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl BuildIndirectArgsBindings {
    pub fn new(gpu: &GpuContext, resources: &GlobalSortGpuResources) -> Self {
        let bind_group_layout = gpu.create_bind_group_layout(
            "global sort build dispatch indirect bind group layout",
            &[
                // visible_count
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // radix_dispatch_args
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
                // radix_params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // DrawIndirectArgs
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

        let bind_group = Self::make_bind_group(gpu, &bind_group_layout, resources);

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
            "global sort build dispatch indirect bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.pipeline.visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources
                        .pipeline
                        .radix_dispatch_args_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.radix_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources
                        .pipeline
                        .draw_indirect_args_buffer
                        .as_entire_binding(),
                },
            ],
        )
    }
}
