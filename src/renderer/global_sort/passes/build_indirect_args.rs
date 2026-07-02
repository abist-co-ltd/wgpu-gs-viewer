use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::bindings::build_indirect_args::BuildIndirectArgsBindings;

pub struct BuildIndirectArgsPass {
    pipeline: wgpu::ComputePipeline,
}

impl BuildIndirectArgsPass {
    pub fn new(gpu: &GpuContext, bindings: &BuildIndirectArgsBindings) -> Self {
        let shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/global_sort/build_indirect_args.compute.wgsl"
        ));

        let pipeline_layout = gpu.create_pipeline_layout(
            "global sort build dispatch indirect pipeline layout",
            &[Some(&bindings.bind_group_layout)],
        );

        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global sort build dispatch indirect pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, bindings: &BuildIndirectArgsBindings) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Global Sort Build Dispatch Indirect Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        pass.set_bind_group(0, &bindings.bind_group, &[]);

        pass.dispatch_workgroups(1, 1, 1);
    }
}
