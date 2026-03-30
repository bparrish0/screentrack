fn main() {
    // Embed compile timestamp into the binary.
    // We print cargo:rerun-if-changed for a non-existent file so this script
    // always reruns, keeping the timestamp fresh on every build.
    let now = chrono::Local::now();
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        now.format("%Y-%m-%d %H:%M:%S %Z")
    );
    println!("cargo:rerun-if-changed=.force-rebuild");
}
