use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::{
    RADIX_SORT_PASSES, bindings::radix_sort::RadixSortBindings,
    global_sort_pipeline_resources::GlobalSortPipelineResources,
};

pub struct RadixSortPass {
    histogram_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
}

impl RadixSortPass {
    pub fn new(gpu: &GpuContext, bindings: &RadixSortBindings) -> Self {
        let histogram_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/global_sort/radix_hist.compute.wgsl"
        ));

        let scatter_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/global_sort/radix_scatter.compute.wgsl"
        ));

        let histogram_pipeline_layout = gpu.create_pipeline_layout(
            "global sort radix histogram pipeline layout",
            &[Some(&bindings.histogram_bind_group_layout)],
        );

        let scatter_pipeline_layout = gpu.create_pipeline_layout(
            "global sort radix scatter pipeline layout",
            &[Some(&bindings.scatter_bind_group_layout)],
        );

        let histogram_pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global sort radix histogram pipeline"),
            layout: Some(&histogram_pipeline_layout),
            module: &histogram_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global sort radix scatter pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            histogram_pipeline,
            scatter_pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &GlobalSortPipelineResources,
        bindings: &RadixSortBindings,
    ) {
        for pass_index in 0..RADIX_SORT_PASSES {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Global Sort Radix Histogram Pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.histogram_pipeline);

                pass.set_bind_group(0, &bindings.histogram_bind_groups[pass_index as usize], &[]);

                pass.dispatch_workgroups_indirect(
                    &pipeline_resources.radix_dispatch_args_buffer,
                    0,
                );
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Global Sort Radix Scatter Pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.scatter_pipeline);

                pass.set_bind_group(0, &bindings.scatter_bind_groups[pass_index as usize], &[]);

                pass.dispatch_workgroups_indirect(
                    &pipeline_resources.radix_dispatch_args_buffer,
                    0,
                );
            }
        }
    }
}
