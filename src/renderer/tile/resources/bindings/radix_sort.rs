use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::RADIX_SORT_PASSES;
use crate::renderer::tile::resources::TileGpuResources;
use crate::renderer::tile::resources::tile_pipeline_resources::RadixPassIndex;

pub struct RadixSortBindings {
    pub build_radix_args_bind_group_layout: wgpu::BindGroupLayout,
    pub build_radix_args_bind_group: wgpu::BindGroup,

    pub radix_histogram_bind_group_layout: wgpu::BindGroupLayout,
    pub radix_scatter_bind_group_layout: wgpu::BindGroupLayout,

    pub radix_histogram_bind_groups: Vec<wgpu::BindGroup>,
    pub radix_scatter_bind_groups: Vec<wgpu::BindGroup>,

    pub radix_pass_index_buffers: Vec<wgpu::Buffer>,
}

impl RadixSortBindings {
    pub fn new(gpu: &GpuContext, resources: &TileGpuResources) -> Self {
        let radix_pass_index_buffers: Vec<wgpu::Buffer> = (0..RADIX_SORT_PASSES)
            .map(|i| {
                gpu.create_buffer_init(
                    "Radix Pass Index Buffer",
                    bytemuck::bytes_of(&RadixPassIndex { value: i as u32 }),
                    wgpu::BufferUsages::UNIFORM,
                )
            })
            .collect();

        let build_radix_args_bind_group_layout = gpu.create_bind_group_layout(
            "build radix args bind group layout",
            &[
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
        );

        let build_radix_args_bind_group = Self::make_build_radix_args_bind_group(
            gpu,
            &build_radix_args_bind_group_layout,
            resources,
        );

        let radix_histogram_bind_group_layout = gpu.create_bind_group_layout(
            "radix histogram bind group layout",
            &[
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
        );

        let radix_scatter_bind_group_layout = gpu.create_bind_group_layout(
            "radix scatter bind group layout",
            &[
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
        );

        let (radix_histogram_bind_groups, radix_scatter_bind_groups) =
            Self::make_per_pass_bind_groups(
                gpu,
                &radix_histogram_bind_group_layout,
                &radix_scatter_bind_group_layout,
                &radix_pass_index_buffers,
                resources,
            );

        Self {
            build_radix_args_bind_group_layout,
            build_radix_args_bind_group,

            radix_histogram_bind_group_layout,
            radix_scatter_bind_group_layout,

            radix_histogram_bind_groups,
            radix_scatter_bind_groups,

            radix_pass_index_buffers,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &TileGpuResources) {
        self.build_radix_args_bind_group = Self::make_build_radix_args_bind_group(
            gpu,
            &self.build_radix_args_bind_group_layout,
            resources,
        );

        let (histogram, scatter) = Self::make_per_pass_bind_groups(
            gpu,
            &self.radix_histogram_bind_group_layout,
            &self.radix_scatter_bind_group_layout,
            &self.radix_pass_index_buffers,
            resources,
        );
        self.radix_histogram_bind_groups = histogram;
        self.radix_scatter_bind_groups = scatter;
    }

    fn make_build_radix_args_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "build radix args bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.pipeline.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.radix_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources
                        .pipeline
                        .radix_dispatch_args_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources
                        .pipeline
                        .tile_range_dispatch_args_buffer
                        .as_entire_binding(),
                },
            ],
        )
    }

    fn make_per_pass_bind_groups(
        gpu: &GpuContext,
        histogram_layout: &wgpu::BindGroupLayout,
        scatter_layout: &wgpu::BindGroupLayout,
        pass_index_buffers: &[wgpu::Buffer],
        resources: &TileGpuResources,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut histogram_bgs = Vec::with_capacity(RADIX_SORT_PASSES as usize);
        let mut scatter_bgs = Vec::with_capacity(RADIX_SORT_PASSES as usize);

        for pass_i in 0..RADIX_SORT_PASSES {
            let even = pass_i % 2 == 0;
            let keys_in = if even {
                &resources.pipeline.pair_keys_buffer
            } else {
                &resources.pipeline.pair_keys_tmp_buffer
            };
            let keys_out = if even {
                &resources.pipeline.pair_keys_tmp_buffer
            } else {
                &resources.pipeline.pair_keys_buffer
            };
            let values_in = if even {
                &resources.pipeline.pair_values_buffer
            } else {
                &resources.pipeline.pair_values_tmp_buffer
            };
            let values_out = if even {
                &resources.pipeline.pair_values_tmp_buffer
            } else {
                &resources.pipeline.pair_values_buffer
            };

            histogram_bgs.push(
                gpu.create_bind_group(
                    "radix histogram bind group",
                    histogram_layout,
                    &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pass_index_buffers[pass_i as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: resources.pipeline.radix_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: keys_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: resources
                                .pipeline
                                .radix_histograms_buffer
                                .as_entire_binding(),
                        },
                    ],
                ),
            );

            scatter_bgs.push(
                gpu.create_bind_group(
                    "radix scatter bind group",
                    scatter_layout,
                    &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pass_index_buffers[pass_i as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: resources.pipeline.radix_params_buffer.as_entire_binding(),
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
                            resource: resources
                                .pipeline
                                .radix_histograms_buffer
                                .as_entire_binding(),
                        },
                    ],
                ),
            );
        }

        (histogram_bgs, scatter_bgs)
    }
}
