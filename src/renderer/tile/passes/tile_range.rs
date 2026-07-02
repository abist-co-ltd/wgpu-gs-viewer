use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::{
    bindings::tile_range::TileRangeBindings, tile_pipeline_resources::TilePipelineResources,
};

pub struct TileRangePass {
    clear_tile_ranges_pipeline: wgpu::ComputePipeline,
    tile_range_pipeline: wgpu::ComputePipeline,
}

impl TileRangePass {
    pub fn new(gpu: &GpuContext, bindings: &TileRangeBindings) -> Self {
        let clear_tile_ranges_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/clear_tile_ranges.compute.wgsl"
        ));
        let clear_tile_ranges_pipeline =
            gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("clear tile ranges pipeline"),
                layout: Some(&gpu.create_pipeline_layout(
                    "clear tile ranges pipeline layout",
                    &[Some(&bindings.clear_tile_ranges_bind_group_layout)],
                )),
                module: &clear_tile_ranges_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let tile_range_shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/tile_range.compute.wgsl"
        ));
        let tile_range_pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile range pipeline"),
            layout: Some(&gpu.create_pipeline_layout(
                "tile range pipeline layout",
                &[Some(&bindings.tile_range_bind_group_layout)],
            )),
            module: &tile_range_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            clear_tile_ranges_pipeline,
            tile_range_pipeline,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &TilePipelineResources,
        bindings: &TileRangeBindings,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Tile Ranges Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_tile_ranges_pipeline);
            pass.set_bind_group(0, &bindings.clear_tile_ranges_bind_group, &[]);
            let workgroup_x = pipeline_resources.tile_count.div_ceil(256);
            pass.dispatch_workgroups(workgroup_x, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Range Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_range_pipeline);
            pass.set_bind_group(0, &bindings.tile_range_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &pipeline_resources.tile_range_dispatch_args_buffer,
                0,
            );
        }
    }
}
