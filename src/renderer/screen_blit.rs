use crate::gpu::{context::GpuContext, frame_targets::FrameTargets};

pub struct ScreenBlitPass {
    pipeline: wgpu::RenderPipeline,
}

impl ScreenBlitPass {
    pub fn new(gpu: &GpuContext, bindings: &ScreenBlitBindings) -> Self {
        let pipeline_layout = gpu.create_pipeline_layout(
            "Screen Blit Pipeline Layout",
            &[Some(&bindings.bind_group_layout)],
        );

        let shader = gpu.create_shader_module(wgpu::include_wgsl!("../shaders/screen_blit.wgsl"));
        let pipeline = gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen blit pipeline"),
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
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
        bindings: &ScreenBlitBindings,
        output_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Screen Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub struct ScreenBlitBindings {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl ScreenBlitBindings {
    pub fn new(gpu: &GpuContext, frame_targets: &FrameTargets) -> Self {
        let bind_group_layout = gpu.create_bind_group_layout(
            "screen blit bind group layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        );

        let bind_group = Self::make_bind_group(
            gpu,
            &bind_group_layout,
            &frame_targets.render_texture_view,
            &frame_targets.render_texture_sampler,
        );

        Self {
            bind_group_layout,
            bind_group,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, frame_targets: &FrameTargets) {
        self.bind_group = Self::make_bind_group(
            gpu,
            &self.bind_group_layout,
            &frame_targets.render_texture_view,
            &frame_targets.render_texture_sampler,
        );
    }

    fn make_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        render_texture_view: &wgpu::TextureView,
        render_texture_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "screen blit bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(render_texture_sampler),
                },
            ],
        )
    }
}
