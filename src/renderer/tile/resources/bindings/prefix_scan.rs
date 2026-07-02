use crate::gpu::context::GpuContext;
use crate::renderer::tile::resources::TileGpuResources;
use crate::renderer::tile::resources::tile_pipeline_resources::{
    PairCountParams, PrefixLevelParams,
};

pub struct PrefixScanBindings {
    pub build_dispatch_args_bind_group_layout: wgpu::BindGroupLayout,
    pub build_dispatch_args_bind_group: wgpu::BindGroup,

    pub scan_bind_group_layout: wgpu::BindGroupLayout,
    pub scan0_bind_group: wgpu::BindGroup,
    pub scan1_bind_group: wgpu::BindGroup,
    pub scan2_bind_group: wgpu::BindGroup,

    pub add_bind_group_layout: wgpu::BindGroupLayout,
    pub add1_bind_group: wgpu::BindGroup,
    pub add0_bind_group: wgpu::BindGroup,

    pub build_total_pairs_bind_group_layout: wgpu::BindGroupLayout,
    pub build_total_pairs_bind_group: wgpu::BindGroup,

    prefix_params0_buffer: wgpu::Buffer,
    prefix_params1_buffer: wgpu::Buffer,
    prefix_params2_buffer: wgpu::Buffer,
    pair_count_params_buffer: wgpu::Buffer,
}

impl PrefixScanBindings {
    pub fn new(gpu: &GpuContext, resources: &TileGpuResources) -> Self {
        let prefix_params0_buffer = gpu.create_buffer_init(
            "Prefix Params Level 0",
            bytemuck::bytes_of(&PrefixLevelParams { level: 0 }),
            wgpu::BufferUsages::UNIFORM,
        );
        let prefix_params1_buffer = gpu.create_buffer_init(
            "Prefix Params Level 1",
            bytemuck::bytes_of(&PrefixLevelParams { level: 1 }),
            wgpu::BufferUsages::UNIFORM,
        );
        let prefix_params2_buffer = gpu.create_buffer_init(
            "Prefix Params Level 2",
            bytemuck::bytes_of(&PrefixLevelParams { level: 2 }),
            wgpu::BufferUsages::UNIFORM,
        );
        let pair_count_params_buffer = gpu.create_buffer_init(
            "Pair Count Params Buffer",
            bytemuck::bytes_of(&PairCountParams {
                max_pairs: resources.pipeline.max_pairs,
            }),
            wgpu::BufferUsages::UNIFORM,
        );

        let build_dispatch_args_bind_group_layout = gpu.create_bind_group_layout(
            "build dispatch args bind group layout",
            &[
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
        );

        let build_dispatch_args_bind_group = Self::make_build_dispatch_args_bind_group(
            gpu,
            &build_dispatch_args_bind_group_layout,
            resources,
        );

        let scan_bind_group_layout = gpu.create_bind_group_layout(
            "scan exclusive bind group layout",
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let (scan0_bind_group, scan1_bind_group, scan2_bind_group) = Self::make_scan_bind_groups(
            gpu,
            &scan_bind_group_layout,
            &prefix_params0_buffer,
            &prefix_params1_buffer,
            &prefix_params2_buffer,
            resources,
        );

        let add_bind_group_layout = gpu.create_bind_group_layout(
            "add block offsets bind group layout",
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
        );

        let (add1_bind_group, add0_bind_group) = Self::make_add_bind_groups(
            gpu,
            &add_bind_group_layout,
            &prefix_params0_buffer,
            &prefix_params1_buffer,
            resources,
        );

        let build_total_pairs_bind_group_layout = gpu.create_bind_group_layout(
            "build total pairs bind group layout",
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
        );

        let build_total_pairs_bind_group = Self::make_build_total_pairs_bind_group(
            gpu,
            &build_total_pairs_bind_group_layout,
            resources,
            &pair_count_params_buffer,
        );

        Self {
            build_dispatch_args_bind_group_layout,
            build_dispatch_args_bind_group,
            scan_bind_group_layout,
            scan0_bind_group,
            scan1_bind_group,
            scan2_bind_group,
            add_bind_group_layout,
            add1_bind_group,
            add0_bind_group,
            build_total_pairs_bind_group_layout,
            build_total_pairs_bind_group,
            prefix_params0_buffer,
            prefix_params1_buffer,
            prefix_params2_buffer,
            pair_count_params_buffer,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &TileGpuResources) {
        self.pair_count_params_buffer = gpu.create_buffer_init(
            "Pair Count Params Buffer",
            bytemuck::bytes_of(&PairCountParams {
                max_pairs: resources.pipeline.max_pairs,
            }),
            wgpu::BufferUsages::UNIFORM,
        );

        self.build_dispatch_args_bind_group = Self::make_build_dispatch_args_bind_group(
            gpu,
            &self.build_dispatch_args_bind_group_layout,
            resources,
        );

        let (scan0, scan1, scan2) = Self::make_scan_bind_groups(
            gpu,
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
            gpu,
            &self.add_bind_group_layout,
            &self.prefix_params0_buffer,
            &self.prefix_params1_buffer,
            resources,
        );
        self.add1_bind_group = add1;
        self.add0_bind_group = add0;

        self.build_total_pairs_bind_group = Self::make_build_total_pairs_bind_group(
            gpu,
            &self.build_total_pairs_bind_group_layout,
            resources,
            &self.pair_count_params_buffer,
        );
    }

    fn make_build_dispatch_args_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        resources: &TileGpuResources,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "build dispatch args bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.pipeline.visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources
                        .pipeline
                        .prefix_dispatch_args_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
            ],
        )
    }

    fn make_scan_bind_groups(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        prefix_params0: &wgpu::Buffer,
        prefix_params1: &wgpu::Buffer,
        prefix_params2: &wgpu::Buffer,
        resources: &TileGpuResources,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let scan0 = gpu.create_bind_group(
            "scan0 bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.tiles_touched_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.pipeline.block_sums0_buffer.as_entire_binding(),
                },
            ],
        );
        let scan1 = gpu.create_bind_group(
            "scan1 bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.block_sums0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.block_offsets0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.pipeline.block_sums1_buffer.as_entire_binding(),
                },
            ],
        );
        let scan2 = gpu.create_bind_group(
            "scan2 bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.block_sums1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.block_offsets1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.pipeline.block_sums2_buffer.as_entire_binding(),
                },
            ],
        );
        (scan0, scan1, scan2)
    }

    fn make_add_bind_groups(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        prefix_params0: &wgpu::Buffer,
        prefix_params1: &wgpu::Buffer,
        resources: &TileGpuResources,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let add1 = gpu.create_bind_group(
            "add1 bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.block_offsets0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.block_offsets1_buffer.as_entire_binding(),
                },
            ],
        );
        let add0 = gpu.create_bind_group(
            "add0 bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_params0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.pipeline.block_offsets0_buffer.as_entire_binding(),
                },
            ],
        );
        (add1, add0)
    }

    fn make_build_total_pairs_bind_group(
        gpu: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        resources: &TileGpuResources,
        pair_count_params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            "build total pairs bind group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.pipeline.prefix_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.pipeline.block_sums1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.pipeline.total_pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_count_params_buffer.as_entire_binding(),
                },
            ],
        )
    }
}
