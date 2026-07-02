use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::{
    bindings::render::RenderBindings, global_sort_pipeline_resources::GlobalSortPipelineResources,
};

pub struct RenderPass {
    pipeline: wgpu::RenderPipeline,
}

impl RenderPass {
    pub fn new(gpu: &GpuContext, bindings: &RenderBindings) -> Self {
        let shader = gpu.create_shader_module(wgpu::include_wgsl!(
            "../../../shaders/global_sort/render.wgsl"
        ));

        let pipeline_layout = gpu.create_pipeline_layout(
            "Global Sort Render Pipeline Layout",
            &[Some(&bindings.bind_group_layout)],
        );

        let pipeline = gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Global Sort Render Pipeline"),

            layout: Some(&pipeline_layout),

            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),

                buffers: &[],

                compilation_options: Default::default(),
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),

                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },

            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_resources: &GlobalSortPipelineResources,
        bindings: &RenderBindings,
        target_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Global Sort Render Pass"),

            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),

                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],

            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);

        pass.set_bind_group(0, &bindings.bind_group, &[]);

        pass.draw_indirect(&pipeline_resources.draw_indirect_args_buffer, 0);
    }
}
