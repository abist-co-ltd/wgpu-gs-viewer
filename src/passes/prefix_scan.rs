use wgpu::util::DeviceExt;

use crate::gaussian_resources::{
    DispatchIndirectArgs, GaussianResources, PairCountParams, PrefixLevelParams,
};

pub struct PrefixScanPass {
    build_dispatch_args_bind_group_layout: wgpu::BindGroupLayout,
    build_dispatch_args_bind_group: wgpu::BindGroup,
    build_dispatch_args_pipeline: wgpu::ComputePipeline,

    scan_bind_group_layout: wgpu::BindGroupLayout,
    scan_exclusive_pipeline: wgpu::ComputePipeline,
    scan0_bind_group: wgpu::BindGroup,
    scan1_bind_group: wgpu::BindGroup,
    scan2_bind_group: wgpu::BindGroup,

    add_bind_group_layout: wgpu::BindGroupLayout,
    add_block_offsets_pipeline: wgpu::ComputePipeline,
    add1_bind_group: wgpu::BindGroup,
    add0_bind_group: wgpu::BindGroup,

    build_total_pairs_bind_group_layout: wgpu::BindGroupLayout,
    build_total_pairs_bind_group: wgpu::BindGroup,
    build_total_pairs_pipeline: wgpu::ComputePipeline,

    prefix_params0_buffer: wgpu::Buffer,
    prefix_params1_buffer: wgpu::Buffer,
    prefix_params2_buffer: wgpu::Buffer,
    pair_count_params_buffer: wgpu::Buffer,
}

impl PrefixScanPass {
    pub fn new(device: &wgpu::Device, resources: &GaussianResources) -> Self {
        let prefix_params0_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Prefix Params Level 0"),
            contents: bytemuck::bytes_of(&PrefixLevelParams { level: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let prefix_params1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Prefix Params Level 1"),
            contents: bytemuck::bytes_of(&PrefixLevelParams { level: 1 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let prefix_params2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Prefix Params Level 2"),
            contents: bytemuck::bytes_of(&PrefixLevelParams { level: 2 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pair_count_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pair Count Params Buffer"),
                contents: bytemuck::bytes_of(&PairCountParams {
                    max_pairs: resources.max_pairs,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let build_dispatch_args_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build dispatch args bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                ],
            });

        let build_dispatch_args_bind_group = Self::make_build_dispatch_args_bind_group(
            device,
            &build_dispatch_args_bind_group_layout,
            resources,
        );

        let build_dispatch_args_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/build_dispatch_args.compute.wgsl"
        ));
        let build_dispatch_args_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("build dispatch args pipeline layout"),
                bind_group_layouts: &[Some(&build_dispatch_args_bind_group_layout)],
                immediate_size: 0,
            });
        let build_dispatch_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build dispatch args pipeline"),
                layout: Some(&build_dispatch_args_pipeline_layout),
                module: &build_dispatch_args_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let scan_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scan exclusive bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let (scan0_bind_group, scan1_bind_group, scan2_bind_group) = Self::make_scan_bind_groups(
            device,
            &scan_bind_group_layout,
            &prefix_params0_buffer,
            &prefix_params1_buffer,
            &prefix_params2_buffer,
            resources,
        );

        let scan_exclusive_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/scan_exclusive_level.compute.wgsl"
        ));
        let scan_exclusive_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scan exclusive pipeline layout"),
                bind_group_layouts: &[Some(&scan_bind_group_layout)],
                immediate_size: 0,
            });
        let scan_exclusive_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scan exclusive pipeline"),
                layout: Some(&scan_exclusive_pipeline_layout),
                module: &scan_exclusive_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let add_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("add block offsets bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let (add1_bind_group, add0_bind_group) = Self::make_add_bind_groups(
            device,
            &add_bind_group_layout,
            &prefix_params0_buffer,
            &prefix_params1_buffer,
            resources,
        );

        let add_block_offsets_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/add_block_offsets.compute.wgsl"
        ));
        let add_block_offsets_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("add block offsets pipeline layout"),
                bind_group_layouts: &[Some(&add_bind_group_layout)],
                immediate_size: 0,
            });
        let add_block_offsets_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("add block offsets pipeline"),
                layout: Some(&add_block_offsets_pipeline_layout),
                module: &add_block_offsets_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let build_total_pairs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build total pairs bind group layout"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let build_total_pairs_bind_group = Self::make_build_total_pairs_bind_group(
            device,
            &build_total_pairs_bind_group_layout,
            resources,
            &pair_count_params_buffer,
        );

        let build_total_pairs_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/build_total_pairs.compute.wgsl"
        ));
        let build_total_pairs_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("build total pairs pipeline layout"),
                bind_group_layouts: &[Some(&build_total_pairs_bind_group_layout)],
                immediate_size: 0,
            });
        let build_total_pairs_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build total pairs pipeline"),
                layout: Some(&build_total_pairs_pipeline_layout),
                module: &build_total_pairs_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            build_dispatch_args_bind_group_layout,
            build_dispatch_args_bind_group,
            build_dispatch_args_pipeline,
            scan_bind_group_layout,
            scan_exclusive_pipeline,
            scan0_bind_group,
            scan1_bind_group,
            scan2_bind_group,
            add_bind_group_layout,
            add_block_offsets_pipeline,
            add1_bind_group,
            add0_bind_group,
            build_total_pairs_bind_group_layout,
            build_total_pairs_bind_group,
            build_total_pairs_pipeline,
            prefix_params0_buffer,
            prefix_params1_buffer,
            prefix_params2_buffer,
            pair_count_params_buffer,
        }
    }

    pub fn recreate_bind_group(&mut self, device: &wgpu::Device, resources: &GaussianResources) {
        self.pair_count_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pair Count Params Buffer"),
                contents: bytemuck::bytes_of(&PairCountParams {
                    max_pairs: resources.max_pairs,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.build_dispatch_args_bind_group = Self::make_build_dispatch_args_bind_group(
            device,
            &self.build_dispatch_args_bind_group_layout,
            resources,
        );

        let (scan0, scan1, scan2) = Self::make_scan_bind_groups(
            device,
            &self.scan_bind_group_layout,
            &self.prefix_params0_buffer,
            &self.prefix_params1_buffer,
            &self.prefix_params2_buffer,
            resources,
        );
        self.scan0_bind_group = scan0;
        self.scan1_bind_group = scan1;
        self.scan2_bind_group = scan2;

        let (add1, add0) = Self::make_add_bind_groups(
            device,
            &self.add_bind_group_layout,
            &self.prefix_params0_buffer,
            &self.prefix_params1_buffer,
            resources,
        );
        self.add1_bind_group = add1;
        self.add0_bind_group = add0;

        self.build_total_pairs_bind_group = Self::make_build_total_pairs_bind_group(
            device,
            &self.build_total_pairs_bind_group_layout,
            resources,
            &self.pair_count_params_buffer,
        );
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, resources: &GaussianResources) {
        const DISPATCH_ARGS_SIZE: u64 = std::mem::size_of::<DispatchIndirectArgs>() as u64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Dispatch Args Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_dispatch_args_pipeline);
            pass.set_bind_group(0, &self.build_dispatch_args_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &self.scan0_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&resources.prefix_dispatch_args_buffer, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &self.scan1_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Scan Level 2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_exclusive_pipeline);
            pass.set_bind_group(0, &self.scan2_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 2,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Add Level 1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_block_offsets_pipeline);
            pass.set_bind_group(0, &self.add1_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 3,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prefix Add Level 0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_block_offsets_pipeline);
            pass.set_bind_group(0, &self.add0_bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &resources.prefix_dispatch_args_buffer,
                DISPATCH_ARGS_SIZE * 4,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Total Pairs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_total_pairs_pipeline);
            pass.set_bind_group(0, &self.build_total_pairs_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    fn make_build_dispatch_args_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        resources: &GaussianResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build dispatch args bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_dispatch_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn make_scan_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        prefix_params0: &wgpu::Buffer,
        prefix_params1: &wgpu::Buffer,
        prefix_params2: &wgpu::Buffer,
        resources: &GaussianResources,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let scan0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan0 bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.tiles_touched_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.block_sums0_buffer.as_entire_binding(),
                },
            ],
        });
        let scan1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan1 bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.block_sums0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.block_offsets0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.block_sums1_buffer.as_entire_binding(),
                },
            ],
        });
        let scan2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan2 bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.block_sums1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.block_offsets1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.block_sums2_buffer.as_entire_binding(),
                },
            ],
        });
        (scan0, scan1, scan2)
    }

    fn make_add_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        prefix_params0: &wgpu::Buffer,
        prefix_params1: &wgpu::Buffer,
        resources: &GaussianResources,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let add1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add1 bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.block_offsets0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.block_offsets1_buffer.as_entire_binding(),
                },
            ],
        });
        let add0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add0 bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.block_offsets0_buffer.as_entire_binding(),
                },
            ],
        });
        (add1, add0)
    }

    fn make_build_total_pairs_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        resources: &GaussianResources,
        pair_count_params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build total pairs bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.block_sums1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_count_params_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
