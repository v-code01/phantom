use crate::device;

/// MSL kernel: reads buf[0] (uint), writes buf[0] * 2.
/// One thread, one threadgroup — proves zero-copy UMA semantics, not occupancy.
const DOUBLE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void double_value(
    device uint* buf [[buffer(0)]],
    uint          id [[thread_position_in_grid]]
) {
    if (id == 0) {
        buf[0] = buf[0] * 2u;
    }
}
"#;

/// Writes `value` into a shared MTLBuffer, dispatches a GPU kernel that doubles it,
/// waits for completion, and returns the result — no CPU↔GPU copy at any point.
///
/// # Panics
/// Panics if no Metal device is available, if shader compilation fails, or if
/// command buffer execution fails.
pub fn double_on_gpu(value: u32) -> u32 {
    let dev = device::system_device();

    let opts = metal::CompileOptions::new();
    let lib = dev
        .new_library_with_source(DOUBLE_SHADER, &opts)
        .expect("Metal shader compilation failed");
    let func = lib
        .get_function("double_value", None)
        .expect("'double_value' function not found in compiled library");
    let pipeline = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("failed to create compute pipeline state");

    let buf = device::shared_buffer(&dev, 4);
    // SAFETY: StorageModeShared guarantees non-null contents pointer. Metal guarantees
    // page-aligned allocation satisfying *mut u32 alignment. `buf` outlives `ptr`
    // (same scope). No GPU encoder is active yet so there is no concurrent GPU writer.
    let ptr = buf.contents() as *mut u32;
    unsafe { *ptr = value; }

    let queue = dev.new_command_queue();
    let cmd   = queue.new_command_buffer();
    let enc   = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    let grid = metal::MTLSize { width: 1, height: 1, depth: 1 };
    let tg   = metal::MTLSize { width: 1, height: 1, depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // SAFETY: cmd.wait_until_completed() ensures the GPU has finished writing buf[0].
    // No other writer exists. Alignment guaranteed by Metal allocator.
    unsafe { *(buf.contents() as *const u32) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_doubles_cpu_written_value_without_copy() {
        let result = double_on_gpu(21);
        assert_eq!(result, 42,
            "GPU must double the value written by CPU into shared MTLBuffer");
    }

    #[test]
    fn gpu_handles_zero() {
        let result = double_on_gpu(0);
        assert_eq!(result, 0);
    }

    #[test]
    fn gpu_handles_large_value() {
        let result = double_on_gpu(1_000_000);
        assert_eq!(result, 2_000_000);
    }
}
