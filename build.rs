extern crate cc;


fn main() {
    cc::Build::new()
        .file("src/axpb.c")
	.compile("libaxpbc.a");

    cc::Build::new()
        .file("src/axpb.cpp")
        .cpp(true)
        .compile("libaxpbcpp.a");


    #[cfg(feature="gpu")] {
	cc::Build::new()
            .cpp(true)
            .cuda(true)
            .flag("-cudart=shared")
            .flag("-gencode")
            .flag("arch=compute_61,code=sm_61")
            .file("src/axpb.cu")
            .compile("libaxpbcu.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    }
}
