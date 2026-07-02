use crate::gpu::context::GpuContext;
use crate::renderer::global_sort::resources::{
    GlobalSortGpuResources, RADIX_SORT_PASSES, global_sort_pipeline_resources::RadixPassIndex,
};

const RADIX_SORT_BINS: u32 = 256;
const MAX_RADIX_WORKGROUPS: u32 = 256;

pub struct RadixSortBindings {
    pub histogram_bind_group_layout: wgpu::BindGroupLayout,
    pub scatter_bind_group_layout: wgpu::BindGroupLayout,

    pub histogram_bind_groups: Vec<wgpu::BindGroup>,
    pub scatter_bind_groups: Vec<wgpu::BindGroup>,

    radix_histograms_buffer: wgpu::Buffer,
    pass_index_buffers: Vec<wgpu::Buffer>,
}

impl RadixSortBindings {
    pub fn new(gpu: &GpuContext, resources: &GlobalSortGpuResources) -> Self {
        let radix_histogram_len = RADIX_SORT_BINS * MAX_RADIX_WORKGROUPS;
        let radix_histograms_buffer = gpu.create_buffer(
            "Global Sort Radix Histograms Buffer",
            std::mem::size_of::<u32>() as u64 * radix_histogram_len as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let pass_index_buffers = Self::create_pass_index_buffers(gpu);

        let histogram_bind_group_layout = Self::create_histogram_bind_group_layout(gpu);

        let scatter_bind_group_layout = Self::create_scatter_bind_group_layout(gpu);

        let (histogram_bind_groups, scatter_bind_groups) = Self::make_per_pass_bind_groups(
            gpu,
            &histogram_bind_group_layout,
            &scatter_bind_group_layout,
            &radix_histograms_buffer,
            &pass_index_buffers,
            resources,
        );

        Self {
            histogram_bind_group_layout,
            scatter_bind_group_layout,
            histogram_bind_groups,
            scatter_bind_groups,
            radix_histograms_buffer,
            pass_index_buffers,
        }
    }

    pub fn recreate(&mut self, gpu: &GpuContext, resources: &GlobalSortGpuResources) {
        let (histogram_bind_groups, scatter_bind_groups) = Self::make_per_pass_bind_groups(
            gpu,
            &self.histogram_bind_group_layout,
            &self.scatter_bind_group_layout,
            &self.radix_histograms_buffer,
            &self.pass_index_buffers,
            resources,
        );

        self.histogram_bind_groups = histogram_bind_groups;

        self.scatter_bind_groups = scatter_bind_groups;
    }

    fn create_pass_index_buffers(gpu: &GpuContext) -> Vec<wgpu::Buffer> {
        (0..RADIX_SORT_PASSES)
            .map(|pass_index| {
                gpu.create_buffer_init(
                    "Global Sort Radix Pass Index Buffer",
                    bytemuck::bytes_of(&RadixPassIndex {
                        value: pass_index as u32,
                    }),
                    wgpu::BufferUsages::UNIFORM,
                )
            })
            .collect()
    }

    fn create_histogram_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
        gpu.create_bind_group_layout(
            "Global Sort Radix Histogram Bind Group Layout",
            &[
                // RadixPassIndex
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
                // RadixSortParams[]
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
                // keys_in[]
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
                // histograms[]
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
        )
    }

    fn create_scatter_bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
        gpu.create_bind_group_layout(
            "Global Sort Radix Scatter Bind Group Layout",
            &[
                // RadixPassIndex
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
                // RadixSortParams[]
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
                // keys_in[]
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
                // keys_out[]
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
                // values_in[]
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
                // values_out[]
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
                // histograms[]
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
        )
    }

    fn make_per_pass_bind_groups(
        gpu: &GpuContext,
        histogram_layout: &wgpu::BindGroupLayout,
        scatter_layout: &wgpu::BindGroupLayout,
        radix_histograms_buffer: &wgpu::Buffer,
        pass_index_buffers: &[wgpu::Buffer],
        resources: &GlobalSortGpuResources,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut histogram_bind_groups = Vec::with_capacity(RADIX_SORT_PASSES as usize);

        let mut scatter_bind_groups = Vec::with_capacity(RADIX_SORT_PASSES as usize);

        for pass_index in 0..RADIX_SORT_PASSES {
            let even = pass_index % 2 == 0;

            let keys_in = if even {
                &resources.pipeline.sort_keys_buffer
            } else {
                &resources.pipeline.sort_keys_tmp_buffer
            };

            let keys_out = if even {
                &resources.pipeline.sort_keys_tmp_buffer
            } else {
                &resources.pipeline.sort_keys_buffer
            };

            let values_in = if even {
                &resources.pipeline.sort_values_buffer
            } else {
                &resources.pipeline.sort_values_tmp_buffer
            };

            let values_out = if even {
                &resources.pipeline.sort_values_tmp_buffer
            } else {
                &resources.pipeline.sort_values_buffer
            };

            let histogram_bind_group = gpu.create_bind_group(
                "Global Sort Radix Histogram Bind Group",
                histogram_layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pass_index_buffers[pass_index as usize].as_entire_binding(),
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
                        resource: radix_histograms_buffer.as_entire_binding(),
                    },
                ],
            );

            let scatter_bind_group = gpu.create_bind_group(
                "Global Sort Radix Scatter Bind Group",
                scatter_layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pass_index_buffers[pass_index as usize].as_entire_binding(),
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
                        resource: radix_histograms_buffer.as_entire_binding(),
                    },
                ],
            );

            histogram_bind_groups.push(histogram_bind_group);

            scatter_bind_groups.push(scatter_bind_group);
        }

        (histogram_bind_groups, scatter_bind_groups)
    }
}
