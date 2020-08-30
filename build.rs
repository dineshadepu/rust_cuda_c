extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/axpb.c")
	.compile("libaxpbc.a");

    cc::Build::new()
        .file("src/axpb.cpp")
        .cpp(true)
        .compile("libaxpbcpp.a");
}
