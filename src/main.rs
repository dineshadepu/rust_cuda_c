use std::env;
use std::io;
use std::io::prelude::*;
use std::process::exit;

// for cuda
#[cfg(feature="gpu")]
#[macro_use]
extern crate rustacuda;

#[cfg(feature="gpu")]
#[macro_use]
extern crate rustacuda_derive;

#[cfg(feature="gpu")]
extern crate rustacuda_core;

// pub mod mesh;
pub mod axpb;

const USAGE: &str = "
Usage: rust-cuda-c bench
       rust-cuda-c <demo-name> [ options ]
       rust-cuda-c --help
Run parallel algorithms using sequential rust, parallel rust with rayon
and with GPU with RusaCuda.
Benchmarks:
  - axpb : Run axpb
";

fn usage() -> ! {
    let _ = writeln!(&mut io::stderr(), "{}", USAGE);
    exit(1);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        usage();
    }

    let bench_name = &args[1];
    match &bench_name[..] {
        "axpb" => axpb::main(&args[2..]),
        _ => usage(),
    }
}
