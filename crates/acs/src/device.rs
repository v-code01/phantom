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

#[cfg(test)]
mod tests {
    use super::*;

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
