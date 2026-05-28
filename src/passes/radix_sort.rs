use wgpu::util::DeviceExt;

use crate::gaussian_resources::{GaussianResources, RADIX_SORT_PASSES, RadixPassIndex};

pub struct RadixSortPass {
    build_radix_args_bind_group_layout: wgpu::BindGroupLayout,
    build_radix_args_bind_group: wgpu::BindGroup,
    build_radix_args_pipeline: wgpu::ComputePipeline,

    radix_histogram_bind_group_layout: wgpu::BindGroupLayout,
    radix_scatter_bind_group_layout: wgpu::BindGroupLayout,

    radix_histogram_pipeline: wgpu::ComputePipeline,
    radix_scatter_pipeline: wgpu::ComputePipeline,

    radix_histogram_bind_groups: Vec<wgpu::BindGroup>,
    radix_scatter_bind_groups: Vec<wgpu::BindGroup>,

    radix_pass_index_buffers: Vec<wgpu::Buffer>,
}

impl RadixSortPass {
    pub fn new(device: &wgpu::Device, resources: &GaussianResources) -> Self {
        let radix_pass_index_buffers: Vec<wgpu::Buffer> = (0..RADIX_SORT_PASSES)
            .map(|i| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Radix Pass Index Buffer"),
                    contents: bytemuck::bytes_of(&RadixPassIndex { value: i as u32 }),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

        let build_radix_args_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build radix args bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        let build_radix_args_bind_group = Self::make_build_radix_args_bind_group(
            device,
            &build_radix_args_bind_group_layout,
            resources,
        );

        let build_radix_args_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/build_radix_args.compute.wgsl"
        ));
        let build_radix_args_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("build radix args pipeline layout"),
                bind_group_layouts: &[Some(&build_radix_args_bind_group_layout)],
                immediate_size: 0,
            });
        let build_radix_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build radix args pipeline"),
                layout: Some(&build_radix_args_pipeline_layout),
                module: &build_radix_args_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let radix_histogram_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix histogram bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let radix_scatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix scatter bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let radix_histogram_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/radix_hist.compute.wgsl"));
        let radix_scatter_shader = device
            .create_shader_module(wgpu::include_wgsl!("../shaders/radix_scatter.compute.wgsl"));

        let radix_histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix histogram pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("radix histogram pipeline layout"),
                        bind_group_layouts: &[Some(&radix_histogram_bind_group_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &radix_histogram_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let radix_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("radix scatter pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("radix scatter pipeline layout"),
                        bind_group_layouts: &[Some(&radix_scatter_bind_group_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &radix_scatter_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let (radix_histogram_bind_groups, radix_scatter_bind_groups) =
            Self::make_per_pass_bind_groups(
                device,
                &radix_histogram_bind_group_layout,
                &radix_scatter_bind_group_layout,
                &radix_pass_index_buffers,
                resources,
            );

        Self {
            build_radix_args_bind_group_layout,
            build_radix_args_bind_group,
            build_radix_args_pipeline,
            radix_histogram_bind_group_layout,
            radix_scatter_bind_group_layout,
            radix_histogram_pipeline,
            radix_scatter_pipeline,
            radix_histogram_bind_groups,
            radix_scatter_bind_groups,
            radix_pass_index_buffers,
        }
    }

    pub fn recreate_bind_group(&mut self, device: &wgpu::Device, resources: &GaussianResources) {
        self.build_radix_args_bind_group = Self::make_build_radix_args_bind_group(
            device,
            &self.build_radix_args_bind_group_layout,
            resources,
        );

        let (histogram, scatter) = Self::make_per_pass_bind_groups(
            device,
            &self.radix_histogram_bind_group_layout,
            &self.radix_scatter_bind_group_layout,
            &self.radix_pass_index_buffers,
            resources,
        );
        self.radix_histogram_bind_groups = histogram;
        self.radix_scatter_bind_groups = scatter;
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, resources: &GaussianResources) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Radix Args Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_radix_args_pipeline);
            pass.set_bind_group(0, &self.build_radix_args_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        for pass_i in 0..RADIX_SORT_PASSES {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Histogram Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.radix_histogram_pipeline);
                pass.set_bind_group(0, &self.radix_histogram_bind_groups[pass_i], &[]);
                pass.dispatch_workgroups_indirect(&resources.radix_dispatch_args_buffer, 0);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Scatter Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.radix_scatter_pipeline);
                pass.set_bind_group(0, &self.radix_scatter_bind_groups[pass_i], &[]);
                pass.dispatch_workgroups_indirect(&resources.radix_dispatch_args_buffer, 0);
            }
        }
    }

    fn make_build_radix_args_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        resources: &GaussianResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build radix args bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.radix_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.radix_dispatch_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources
                        .tile_range_dispatch_args_buffer
                        .as_entire_binding(),
                },
            ],
        })
    }

    fn make_per_pass_bind_groups(
        device: &wgpu::Device,
        histogram_layout: &wgpu::BindGroupLayout,
        scatter_layout: &wgpu::BindGroupLayout,
        pass_index_buffers: &[wgpu::Buffer],
        resources: &GaussianResources,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut histogram_bgs = Vec::with_capacity(RADIX_SORT_PASSES);
        let mut scatter_bgs = Vec::with_capacity(RADIX_SORT_PASSES);

        for pass_i in 0..RADIX_SORT_PASSES {
            let even = pass_i % 2 == 0;
            let keys_in = if even {
                &resources.pair_keys_buffer
            } else {
                &resources.pair_keys_tmp_buffer
            };
            let keys_out = if even {
                &resources.pair_keys_tmp_buffer
            } else {
                &resources.pair_keys_buffer
            };
            let values_in = if even {
                &resources.pair_values_buffer
            } else {
                &resources.pair_values_tmp_buffer
            };
            let values_out = if even {
                &resources.pair_values_tmp_buffer
            } else {
                &resources.pair_values_buffer
            };

            histogram_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix histogram bind group"),
                layout: histogram_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pass_index_buffers[pass_i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: resources.radix_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: resources.radix_histograms_buffer.as_entire_binding(),
                    },
                ],
            }));

            scatter_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix scatter bind group"),
                layout: scatter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pass_index_buffers[pass_i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: resources.radix_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: values_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: values_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: resources.radix_histograms_buffer.as_entire_binding(),
                    },
                ],
            }));
        }

        (histogram_bgs, scatter_bgs)
    }
}
