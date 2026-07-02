use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::{
    bindings::preprocess::PreprocessBindings,
    global_sort_pipeline_resources::GlobalSortPipelineResources,
};
use crate::scene::SceneType;

const PREPROCESS_WORKGROUP_SIZE: u32 = 256;

pub struct PreprocessPass {
    pipeline: wgpu::ComputePipeline,
}

impl PreprocessPass {
    pub fn new(gpu: &GpuContext, bindings: &PreprocessBindings, scene_type: SceneType) -> Self {
        let shader = match scene_type {
            SceneType::Gaussian3d => gpu.create_shader_module(wgpu::include_wgsl!(
                "../../../shaders/global_sort/preprocess_3d.compute.wgsl"
            )),
            SceneType::Gaussian4d => gpu.create_shader_module(wgpu::include_wgsl!(
                "../../../shaders/global_sort/preprocess_4d.compute.wgsl"
            )),
        };

        let pipeline_layout = gpu.create_pipeline_layout(
            "preprocess pipeline layout",
            &[Some(&bindings.bind_group_layout)],
        );

        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
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
        pipeline_resources: &GlobalSortPipelineResources,
        bindings: &PreprocessBindings,
    ) {
        let gaussian_count = pipeline_resources.gaussian_count;

        if gaussian_count == 0 {
            return;
        }

        encoder.clear_buffer(&pipeline_resources.visible_count_buffer, 0, None);

        let workgroup_count = gaussian_count.div_ceil(PREPROCESS_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Global Preprocess Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        pass.set_bind_group(0, &bindings.bind_group, &[]);

        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
