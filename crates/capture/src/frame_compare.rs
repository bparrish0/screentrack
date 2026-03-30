use image::{imageops::FilterType, DynamicImage, GrayImage};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct FrameCompareConfig {
    pub downscale_factor: u32,
    pub threshold: f64,
    pub hash_early_exit: bool,
}

impl Default for FrameCompareConfig {
    fn default() -> Self {
        Self {
            downscale_factor: 4,
            threshold: 0.02,
            hash_early_exit: true,
        }
    }
}

pub struct FrameComparer {
    config: FrameCompareConfig,
    prev_hash: Option<u64>,
    prev_histogram: Option<[f64; 256]>,
}

impl FrameComparer {
    pub fn new(config: FrameCompareConfig) -> Self {
        Self {
            config,
            prev_hash: None,
            prev_histogram: None,
        }
    }

    /// Compare current frame with previous. Returns 0.0–1.0 difference score.
    /// Returns 1.0 on the first frame (no previous to compare against).
    pub fn compare(&mut self, current: &DynamicImage) -> f64 {
        // Downscale
        let (w, h) = (
            current.width() / self.config.downscale_factor,
            current.height() / self.config.downscale_factor,
        );
        let small = current.resize_exact(w, h, FilterType::Nearest);
        let gray = small.to_luma8();

        // Hash-based early exit
        let hash = Self::compute_hash(&gray);
        if self.config.hash_early_exit {
            if let Some(prev) = self.prev_hash {
                if hash == prev {
                    return 0.0;
                }
            }
        }

        // Histogram comparison (Hellinger distance)
        let histogram = Self::compute_histogram(&gray);
        let diff = match &self.prev_histogram {
            Some(prev) => Self::hellinger_distance(prev, &histogram),
            None => 1.0, // First frame
        };

        self.prev_hash = Some(hash);
        self.prev_histogram = Some(histogram);

        diff
    }

    /// Returns true if the frame has changed enough to warrant capture.
    pub fn has_changed(&mut self, current: &DynamicImage) -> bool {
        self.compare(current) >= self.config.threshold
    }

    pub fn reset(&mut self) {
        self.prev_hash = None;
        self.prev_histogram = None;
    }

    fn compute_hash(image: &GrayImage) -> u64 {
        let mut hasher = DefaultHasher::new();
        image.as_raw().hash(&mut hasher);
        hasher.finish()
    }

    fn compute_histogram(image: &GrayImage) -> [f64; 256] {
        let mut hist = [0u32; 256];
        for pixel in image.pixels() {
            hist[pixel.0[0] as usize] += 1;
        }
        let total = image.pixels().count() as f64;
        let mut normalized = [0.0f64; 256];
        for i in 0..256 {
            normalized[i] = hist[i] as f64 / total;
        }
        normalized
    }

    fn hellinger_distance(a: &[f64; 256], b: &[f64; 256]) -> f64 {
        let mut sum = 0.0;
        for i in 0..256 {
            let diff = a[i].sqrt() - b[i].sqrt();
            sum += diff * diff;
        }
        (sum / 2.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};

    fn solid_image(r: u8, g: u8, b: u8) -> DynamicImage {
        let img = RgbaImage::from_fn(200, 200, |_, _| image::Rgba([r, g, b, 255]));
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_identical_frames() {
        let mut comparer = FrameComparer::new(FrameCompareConfig::default());
        let img = solid_image(100, 100, 100);

        let d1 = comparer.compare(&img); // First frame → 1.0
        assert_eq!(d1, 1.0);

        let d2 = comparer.compare(&img); // Identical → 0.0
        assert_eq!(d2, 0.0);
    }

    #[test]
    fn test_different_frames() {
        let mut comparer = FrameComparer::new(FrameCompareConfig::default());
        let white = solid_image(255, 255, 255);
        let black = solid_image(0, 0, 0);

        comparer.compare(&white);
        let diff = comparer.compare(&black);
        assert!(diff > 0.5, "Expected large diff, got {diff}");
    }

    #[test]
    fn test_has_changed() {
        let mut comparer = FrameComparer::new(FrameCompareConfig::default());
        let img = solid_image(100, 100, 100);

        assert!(comparer.has_changed(&img)); // First frame always changed
        assert!(!comparer.has_changed(&img)); // Same frame
    }
}
