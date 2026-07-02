use crate::{gpu::context::GpuContext, resources::scene::SceneResource};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AxisVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl AxisVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<AxisVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct AxisPass {
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    pipeline: wgpu::RenderPipeline,
}

impl AxisPass {
    pub fn new(gpu: &GpuContext, scene_resource: &SceneResource) -> Self {
        let bind_group_layout = gpu.create_bind_group_layout(
            "axis bind group layout",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        );

        let bind_group =
            Self::make_bind_group(gpu, &bind_group_layout, &scene_resource.uniform_buffer);

        let axis_length = 10000.0;

        let vertices = [
            // X axis: red
            AxisVertex {
                position: [-axis_length, 0.0, 0.0],
                color: [1.0, 0.1, 0.1],
            },
            AxisVertex {
                position: [axis_length, 0.0, 0.0],
                color: [1.0, 0.1, 0.1],
            },
            // Y axis: green
            AxisVertex {
                position: [0.0, -axis_length, 0.0],
                color: [0.1, 1.0, 0.1],
            },
            AxisVertex {
                position: [0.0, axis_length, 0.0],
                color: [0.1, 1.0, 0.1],
            },
            // Z axis: blue
            AxisVertex {
                position: [0.0, 0.0, -axis_length],
                color: [0.1, 0.3, 1.0],
            },
            AxisVertex {
                position: [0.0, 0.0, axis_length],
                color: [0.1, 0.3, 1.0],
            },
        ];

        let vertex_buffer =
            gpu.create_buffer_init("axis vertex buffer", &vertices, wgpu::BufferUsages::VERTEX);

        let shader = gpu.create_shader_module(wgpu::include_wgsl!("../shaders/axis.wgsl"));

        let pipeline_layout =
            gpu.create_pipeline_layout("axis pipeline layout", &[Some(&bind_group_layout)]);

        let pipeline = gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("axis pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[AxisVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            bind_group,
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            pipeline,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Axis Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }

    fn make_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        scene_uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "axis bind group",
            layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform_buffer.as_entire_binding(),
            }],
        )
    }
}
