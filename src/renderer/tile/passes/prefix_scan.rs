use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::{
    bindings::prefix_scan::PrefixScanBindings,
    tile_pipeline_resources::{DispatchIndirectArgs, TilePipelineResources},
};

pub struct PrefixScanPass {
    build_dispatch_args_pipeline: wgpu::ComputePipeline,
    scan_exclusive_pipeline: wgpu::ComputePipeline,
    add_block_offsets_pipeline: wgpu::ComputePipeline,
    build_total_pairs_pipeline: wgpu::ComputePipeline,
}

impl PrefixScanPass {
    pub fn new(gpu: &GpuContext, bindings: &PrefixScanBindings) -> Self {
        let build_dispatch_args_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/build_dispatch_args.compute.wgsl"
        ));
        let build_dispatch_args_pipeline_layout = gpu.create_pipeline_layout(
            "build dispatch args pipeline layout",
            &[Some(&bindings.build_dispatch_args_bind_group_layout)],
        );
        let build_dispatch_args_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build dispatch args pipeline"),
                layout: Some(&build_dispatch_args_pipeline_layout),
                module: &build_dispatch_args_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let scan_exclusive_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/scan_exclusive_level.compute.wgsl"
        ));
        let scan_exclusive_pipeline_layout = gpu.create_pipeline_layout(
            "scan exclusive pipeline layout",
            &[Some(&bindings.scan_bind_group_layout)],
        );
        let scan_exclusive_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scan exclusive pipeline"),
                layout: Some(&scan_exclusive_pipeline_layout),
                module: &scan_exclusive_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let add_block_offsets_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/add_block_offsets.compute.wgsl"
        ));
        let add_block_offsets_pipeline_layout = gpu.create_pipeline_layout(
            "add block offsets pipeline layout",
            &[Some(&bindings.add_bind_group_layout)],
        );
        let add_block_offsets_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("add block offsets pipeline"),
                layout: Some(&add_block_offsets_pipeline_layout),
                module: &add_block_offsets_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let build_total_pairs_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/build_total_pairs.compute.wgsl"
        ));
        let build_total_pairs_pipeline_layout = gpu.create_pipeline_layout(
            "build total pairs pipeline layout",
            &[Some(&bindings.build_total_pairs_bind_group_layout)],
        );
        let build_total_pairs_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build total pairs pipeline"),
                layout: Some(&build_total_pairs_pipeline_layout),
                module: &build_total_pairs_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            build_dispatch_args_pipeline,
            scan_exclusive_pipeline,
            add_block_offsets_pipeline,
            build_total_pairs_pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &TilePipelineResources,
        bindings: &PrefixScanBindings,
    ) {
        const DISPATCH_ARGS_SIZE: u64 = std::mem::size_of::<DispatchIndirectArgs>() as u64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Dispatch Args Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_dispatch_args_pipeline);
            pass.set_bind_group(0, &bindings.build_dispatch_args_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &bindings.scan0_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&pipeline_resources.prefix_dispatch_args_buffer, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &bindings.scan1_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &pipeline_resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &bindings.scan2_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &pipeline_resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 2,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Add Level 1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_block_offsets_pipeline);
            pass.set_bind_group(0, &bindings.add1_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &pipeline_resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 3,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Add Level 0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_block_offsets_pipeline);
            pass.set_bind_group(0, &bindings.add0_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &pipeline_resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 4,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Total Pairs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_total_pairs_pipeline);
            pass.set_bind_group(0, &bindings.build_total_pairs_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
