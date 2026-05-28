use wgpu::util::DeviceExt;

use crate::gaussian_resources::{GaussianResources, TileRangeParams};

pub struct TileRangePass {
    clear_tile_ranges_bind_group_layout: wgpu::BindGroupLayout,
    clear_tile_ranges_bind_group: wgpu::BindGroup,
    clear_tile_ranges_pipeline: wgpu::ComputePipeline,

    tile_range_bind_group_layout: wgpu::BindGroupLayout,
    tile_range_bind_group: wgpu::BindGroup,
    tile_range_pipeline: wgpu::ComputePipeline,

    tile_range_params_buffer: wgpu::Buffer,
}

impl TileRangePass {
    pub fn new(device: &wgpu::Device, resources: &GaussianResources) -> Self {
        let tile_range_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tile Range Params Buffer"),
                contents: bytemuck::bytes_of(&TileRangeParams {
                    tile_count: resources.tile_count,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let clear_tile_ranges_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clear tile ranges bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let clear_tile_ranges_bind_group = Self::make_clear_tile_ranges_bind_group(
            device,
            &clear_tile_ranges_bind_group_layout,
            &tile_range_params_buffer,
            resources,
        );

        let clear_tile_ranges_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/clear_tile_ranges.compute.wgsl"
        ));
        let clear_tile_ranges_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("clear tile ranges pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("clear tile ranges pipeline layout"),
                        bind_group_layouts: &[Some(&clear_tile_ranges_bind_group_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &clear_tile_ranges_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let tile_range_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile range bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let tile_range_bind_group = Self::make_tile_range_bind_group(
            device,
            &tile_range_bind_group_layout,
            &tile_range_params_buffer,
            resources,
        );

        let tile_range_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/tile_range.compute.wgsl"));
        let tile_range_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile range pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("tile range pipeline layout"),
                        bind_group_layouts: &[Some(&tile_range_bind_group_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &tile_range_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            clear_tile_ranges_bind_group_layout,
            clear_tile_ranges_bind_group,
            clear_tile_ranges_pipeline,
            tile_range_bind_group_layout,
            tile_range_bind_group,
            tile_range_pipeline,
            tile_range_params_buffer,
        }
    }

    pub fn recreate_bind_group(&mut self, device: &wgpu::Device, resources: &GaussianResources) {
        self.clear_tile_ranges_bind_group = Self::make_clear_tile_ranges_bind_group(
            device,
            &self.clear_tile_ranges_bind_group_layout,
            &self.tile_range_params_buffer,
            resources,
        );
        self.tile_range_bind_group = Self::make_tile_range_bind_group(
            device,
            &self.tile_range_bind_group_layout,
            &self.tile_range_params_buffer,
            resources,
        );
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, resources: &GaussianResources) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Tile Ranges Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_tile_ranges_pipeline);
            pass.set_bind_group(0, &self.clear_tile_ranges_bind_group, &[]);
            let workgroup_x = resources.tile_count.div_ceil(256);
            pass.dispatch_workgroups(workgroup_x, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Range Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_range_pipeline);
            pass.set_bind_group(0, &self.tile_range_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&resources.tile_range_dispatch_args_buffer, 0);
        }
    }

    fn make_clear_tile_ranges_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        tile_range_params_buffer: &wgpu::Buffer,
        resources: &GaussianResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear tile ranges bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_range_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.tile_ranges_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn make_tile_range_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        tile_range_params_buffer: &wgpu::Buffer,
        resources: &GaussianResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile range bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_range_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pair_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.tile_ranges_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
