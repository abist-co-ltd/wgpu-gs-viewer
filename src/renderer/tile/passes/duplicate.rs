use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::{
    bindings::duplicate::DuplicateBindings, tile_pipeline_resources::TilePipelineResources,
};

pub struct DuplicatePass {
    pipeline: wgpu::ComputePipeline,
}

impl DuplicatePass {
    pub fn new(gpu: &GpuContext, bindings: &DuplicateBindings) -> Self {
        let shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/duplicate.compute.wgsl"
        ));
        let pipeline_layout = gpu.create_pipeline_layout(
            "duplicate pipeline layout",
            &[Some(&bindings.bind_group_layout)],
        );
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("duplicate pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &TilePipelineResources,
        bindings: &DuplicateBindings,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Duplicate Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.bind_group, &[]);
        // args[0] = ceil(visible_count / 256)
        pass.dispatch_workgroups_indirect(&pipeline_resources.prefix_dispatch_args_buffer, 0);
    }
}
