fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    nlprule_build::BinaryBuilder::new(
        &["en"], // Note: putting nothing builds everything
        std::env::var("OUT_DIR").expect("OUT_DIR should be set for `build.rs`"),
    )
    .build()
    .expect("Failed to build the nlprule resource binaries")
    .validate()
    .expect("Invalid nlprule binaries");
}
