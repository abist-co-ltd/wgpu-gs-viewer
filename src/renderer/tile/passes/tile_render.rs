use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::{
    bindings::tile_render::TileRenderBindings, tile_pipeline_resources::TilePipelineResources,
};

pub struct TileRenderPass {
    pipeline: wgpu::ComputePipeline,
}

impl TileRenderPass {
    pub fn new(gpu: &GpuContext, bindings: &TileRenderBindings) -> Self {
        let shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/tile/tile_render.compute.wgsl"
        ));
        let pipeline_layout = gpu.create_pipeline_layout(
            "tile render pipeline layout",
            &[Some(&bindings.bind_group_layout)],
        );
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile render pipeline"),
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
        bindings: &TileRenderBindings,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tile Render Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.bind_group, &[]);
        pass.dispatch_workgroups(
            pipeline_resources.tiles_width,
            pipeline_resources.tiles_height,
            1,
        );
    }
}
