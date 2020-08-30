use rayon::prelude::*;
use structopt::StructOpt;

// // for cuda
// use rustacuda::prelude::*;
// use rustacuda::memory::DeviceBox;
// use std::error::Error;
// use std::ffi::CString;

// from axpb.c file
extern "C" {
    fn double_input(input: libc::c_int) -> libc::c_int;
    fn axpb_c(a: *const libc::c_double, b: *const libc::c_double, c: *mut libc::c_double,
	      n: libc::c_int);

    fn axpb_cpp(a: *const libc::c_double, b: *const libc::c_double, c: *mut libc::c_double,
		n: libc::c_int);
}

fn true_or_false(s: &str) -> Result<bool, &'static str> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("expected `true` or `false`"),
    }
}

/// Search for a pattern in a file and display the lines that contain it.
#[derive(StructOpt)]
struct Cli {
    /// To run code in parallel on CPU with `rayon`
    #[structopt(long, parse(try_from_str))]
    rayon: bool,

    /// To run code in parallel on GPU with `rayon`
    #[structopt(long, parse(try_from_str))]
    cuda: bool,
}

pub fn main(args: &[String]) {
    // if args
    // let args = Cli::from_args();
    serial_axpb();
    serial_c_axpb();
    serial_cpp_axpb();
    // serial_cpp_cuda_axpb();
    rayon_parallel_axpb();
    // gpu_parallel_axpb().unwrap();
}

pub fn serial_axpb() {
    let a = vec![1., 2., 3., 4.];
    let b = vec![1., 2., 3., 4.];
    let mut c = vec![0., 0., 0., 0.];

    for i in 0..a.len() {
        c[i] += a[i] + b[i];
    }
    println!("c in serial is {:?}", c)
}

pub fn rayon_parallel_axpb() {
    let a = vec![1., 2., 3., 4.];
    let b = vec![1., 2., 3., 4.];
    let mut c = vec![0., 0., 0., 0.];

    c.par_iter_mut().enumerate().for_each(|(i, c_i)| {
        *c_i = a[i] + b[i];
    });

    println!("c in parallel rayon is {:?}", c)
}

// fn gpu_parallel_axpb() -> Result<(), Box<dyn Error>> {
//     // Initialize the CUDA API
//     rustacuda::init(CudaFlags::empty())?;

//     // Get the first device
//     let device = Device::get_device(0)?;

//     // Create a context associated to this device
//     // let context = Context::create_and_push(
//     //     ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

//     // Load the module containing the function we want to call
//     let module_data = CString::new(include_str!("../resources/add.ptx"))?;
//     let module = Module::load_from_string(&module_data)?;

//     // Create a stream to submit work to
//     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

//     // Allocate space on the device and copy numbers to it.
//     let mut x = DeviceBox::new(&10.0f32)?;
//     let mut y = DeviceBox::new(&20.0f32)?;
//     let mut result = DeviceBox::new(&0.0f32)?;

//     // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
//     // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
//     unsafe {
//         // Launch the `sum` function with one block containing one thread on the given stream.
//         launch!(module.sum<<<1, 1, 0, stream>>>(
//             x.as_device_ptr(),
//             y.as_device_ptr(),
//             result.as_device_ptr(),
//             1 // Length
//         ))?;
//     }

//     // The kernel launch is asynchronous, so we wait for the kernel to finish executing
//     stream.synchronize()?;

//     // Copy the result back to the host
//     let mut result_host = 0.0f32;
//     result.copy_to(&mut result_host)?;

//     println!("Sum is {}", result_host);

//     Ok(())
// }

pub fn serial_c_axpb() {
    let a = vec![1., 2., 3., 4.];
    let b = vec![1., 2., 3., 4.];
    let mut c = vec![0., 0., 0., 0.];

    unsafe { axpb_c(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), c.len() as i32) };

    println!("c in serial c code ffi is {:?}", c);
}


pub fn serial_cpp_axpb() {
    let a = vec![1., 2., 3., 4.];
    let b = vec![1., 2., 3., 4.];
    let mut c = vec![0., 0., 0., 0.];

    unsafe { axpb_cpp(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), c.len() as i32) };

    println!("c in serial cpp code ffi is {:?}", c);
}
