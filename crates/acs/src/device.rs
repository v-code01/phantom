use metal::Device;

/// Returns the default Metal device.
///
/// # Panics
/// Panics if no Metal device is found, or if the device lacks unified memory.
/// PHANTOM requires Apple Silicon — discrete GPU configurations are not supported.
pub fn system_device() -> Device {
    let dev = Device::system_default()
        .expect("no Metal device — PHANTOM requires Apple Silicon");
    assert!(
        dev.has_unified_memory(),
        "Metal device found but lacks unified memory — discrete GPU not supported"
    );
    dev
}

/// Allocates a Metal buffer in shared storage mode.
/// The returned buffer is directly accessible to both CPU and GPU without any copy.
/// `size_bytes` is rounded up to the device's minimum alignment by the Metal runtime.
pub fn shared_buffer(device: &metal::Device, size_bytes: u64) -> metal::Buffer {
    device.new_buffer(
        size_bytes,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Non-panicking variant of `system_device`.
/// Returns `None` if no Metal device is available or if the device lacks unified memory.
/// Use this for capability probes at startup; use `system_device()` in hot paths.
pub fn try_system_device() -> Option<metal::Device> {
    let dev = metal::Device::system_default()?;
    if dev.has_unified_memory() { Some(dev) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn shared_buffer_cpu_write_read_roundtrip() {
        let device = system_device();
        let buf = shared_buffer(&device, 64);
        let ptr = buf.contents() as *mut u32;
        // SAFETY: Metal guarantees page-aligned allocation for new_buffer, satisfying
        // *mut u32 alignment. `buf` outlives `ptr` (same scope). No GPU command encoder
        // is active so there is no concurrent GPU writer producing a data race.
        unsafe {
            *ptr = 0xDEAD_BEEF;
            assert_eq!(
                *ptr, 0xDEAD_BEEF,
                "CPU must read back what it wrote to a shared MTLBuffer"
            );
        }
    }

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn shared_buffer_is_shared_storage_mode() {
        let device = system_device();
        let buf = shared_buffer(&device, 64);
        // contents() returns non-null only for shared/managed storage
        assert!(
            !buf.contents().is_null(),
            "shared MTLBuffer must have CPU-accessible contents pointer"
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn try_system_device_returns_some_on_apple_silicon() {
        let dev = try_system_device();
        assert!(dev.is_some(), "try_system_device() must return Some on Apple Silicon");
        assert!(
            dev.unwrap().has_unified_memory(),
            "device returned by try_system_device() must have unified memory"
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn metal_device_exists() {
        let device = Device::system_default();
        assert!(device.is_some(), "no Metal device found — is this Apple Silicon?");
    }

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn metal_device_has_unified_memory() {
        let device = Device::system_default().expect("no Metal device");
        assert!(
            device.has_unified_memory(),
            "PHANTOM requires Apple Silicon unified memory"
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon hardware"]
    fn system_device_returns_unified_memory_device() {
        let dev = system_device();
        assert!(dev.has_unified_memory());
    }
}
