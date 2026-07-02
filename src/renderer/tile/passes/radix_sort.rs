use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::RADIX_SORT_PASSES;
use crate::renderer::tile::resources::{
    bindings::radix_sort::RadixSortBindings, tile_pipeline_resources::TilePipelineResources,
};
pub struct RadixSortPass {
    build_radix_args_pipeline: wgpu::ComputePipeline,
    radix_histogram_pipeline: wgpu::ComputePipeline,
    radix_scatter_pipeline: wgpu::ComputePipeline,
}

impl RadixSortPass {
    pub fn new(gpu: &GpuContext, bindings: &RadixSortBindings) -> Self {
        let build_radix_args_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/build_radix_args.compute.wgsl"
        ));
        let build_radix_args_pipeline_layout = gpu.create_pipeline_layout(
            "build radix args pipeline layout",
            &[Some(&bindings.build_radix_args_bind_group_layout)],
        );
        let build_radix_args_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build radix args pipeline"),
                layout: Some(&build_radix_args_pipeline_layout),
                module: &build_radix_args_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let radix_histogram_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/radix_hist.compute.wgsl"
        ));
        let radix_scatter_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/radix_scatter.compute.wgsl"
        ));

        let radix_histogram_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix histogram pipeline"),
                layout: Some(&gpu.create_pipeline_layout(
                    "radix histogram pipeline layout",
                    &[Some(&bindings.radix_histogram_bind_group_layout)],
                )),
                module: &radix_histogram_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let radix_scatter_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix scatter pipeline"),
                layout: Some(&gpu.create_pipeline_layout(
                    "radix scatter pipeline layout",
                    &[Some(&bindings.radix_scatter_bind_group_layout)],
                )),
                module: &radix_scatter_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            build_radix_args_pipeline,
            radix_histogram_pipeline,
            radix_scatter_pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &TilePipelineResources,
        bindings: &RadixSortBindings,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Radix Args Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_radix_args_pipeline);
            pass.set_bind_group(0, &bindings.build_radix_args_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        for pass_i in 0..RADIX_SORT_PASSES {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Histogram Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.radix_histogram_pipeline);
                pass.set_bind_group(
                    0,
                    &bindings.radix_histogram_bind_groups[pass_i as usize],
                    &[],
                );
                pass.dispatch_workgroups_indirect(
                    &pipeline_resources.radix_dispatch_args_buffer,
                    0,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Scatter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.radix_scatter_pipeline);
                pass.set_bind_group(0, &bindings.radix_scatter_bind_groups[pass_i as usize], &[]);
                pass.dispatch_workgroups_indirect(
                    &pipeline_resources.radix_dispatch_args_buffer,
                    0,
                );
            }
        }
    }
}
