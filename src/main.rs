use std::{
    hint::spin_loop,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

mod bcs;

fn bench_all(b: Bencher) {
    b.bench::<bcs::Mutex<bcs::LockFree>>();
    b.bench::<std::sync::Mutex<()>>();
    b.bench::<parking_lot::Mutex<()>>();
    b.bench::<system_lock::Mutex>();
}

trait CriticalSection: Send + Sync + 'static {
    const NAME: &'static str;
    fn new() -> Self;
    fn with(&self, f: impl FnOnce());
}

impl CriticalSection for parking_lot::Mutex<()> {
    const NAME: &'static str = "parking_lot";
    fn new() -> Self {
        Self::new(())
    }
    fn with(&self, f: impl FnOnce()) {
        let _guard = self.lock();
        f()
    }
}

impl CriticalSection for std::sync::Mutex<()> {
    const NAME: &'static str = "std::sync::Mutex";
    fn new() -> Self {
        Self::new(())
    }
    fn with(&self, f: impl FnOnce()) {
        let _guard = self
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f()
    }
}

impl CriticalSection for bcs::Mutex<bcs::LockFree> {
    const NAME: &'static str = "BCS lock-free";
    fn new() -> Self {
        Self::default()
    }
    fn with(&self, f: impl FnOnce()) {
        self.with(f)
    }
}

#[cfg(windows)]
mod system_lock {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn AcquireSRWLockExclusive(p: *mut *mut ());
        pub fn ReleaseSRWLockExclusive(p: *mut *mut ());
    }

    pub struct Mutex(std::cell::UnsafeCell<*mut ()>);
    unsafe impl Send for Mutex {}
    unsafe impl Sync for Mutex {}
    impl super::CriticalSection for Mutex {
        const NAME: &'static str = "SRWLOCK";
        fn new() -> Self {
            Self(std::cell::UnsafeCell::new(std::ptr::null_mut()))
        }
        fn with(&self, f: impl FnOnce()) {
            unsafe {
                AcquireSRWLockExclusive(self.0.get());
                f();
                ReleaseSRWLockExclusive(self.0.get());
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod system_lock {
    unsafe extern "C" {
        pub fn os_unfair_lock_lock(p: *mut u32);
        pub fn os_unfair_lock_unlock(p: *mut u32);
    }

    pub struct Mutex(std::cell::UnsafeCell<u32>);
    unsafe impl Send for Mutex {}
    unsafe impl Sync for Mutex {}
    impl super::CriticalSection for Mutex {
        const NAME: &'static str = "os_unfair_lock";
        fn new() -> Self {
            Self(std::cell::UnsafeCell::new(0))
        }
        fn with(&self, f: impl FnOnce()) {
            unsafe {
                os_unfair_lock_lock(self.0.get());
                f();
                os_unfair_lock_unlock(self.0.get());
            }
        }
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod system_lock {
    pub struct Mutex(std::cell::UnsafeCell<libc::pthread_mutex_t>);
    unsafe impl Send for Mutex {}
    unsafe impl Sync for Mutex {}
    impl super::CriticalSection for Mutex {
        const NAME: &'static str = "pthread_mutex_t";
        fn new() -> Self {
            Self(std::cell::UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER))
        }
        fn with(&self, f: impl FnOnce()) {
            unsafe {
                debug_assert_eq!(libc::pthread_mutex_lock(self.0.get()), 0);
                f();
                debug_assert_eq!(libc::pthread_mutex_unlock(self.0.get()), 0);
            }
        }
    }
}

fn main() {
    let parse_duration = |arg: &str| {
        Some(if arg.ends_with("ns") {
            arg.split("ns").next()?.parse::<u64>().ok()?
        } else if arg.ends_with("us") {
            arg.split("us").next()?.parse::<u64>().ok()? * 1_000
        } else if arg.ends_with("ms") {
            arg.split("ms").next()?.parse::<u64>().ok()? * 1_000_000
        } else if arg.ends_with("s") {
            arg.split("s").next()?.parse::<u64>().ok()? * 1_000_000_000
        } else {
            return None;
        })
    };

    let mut args = std::env::args().skip(1);
    let n_measure = parse_arg(&mut args, "[measure]", false, parse_duration);
    let n_threads = parse_arg(&mut args, "[threads]", true, |x| x.parse::<usize>().ok());
    let n_locked = parse_arg(&mut args, "[locked]", true, parse_duration);
    let n_unlocked = parse_arg(&mut args, "[unlocked]", true, parse_duration);

    let ns_per_work = {
        let num_attempts = 10;
        let attempts = (0..num_attempts)
            .map(|_| {
                let spin_count = 10_000;
                let start = Instant::now();
                (0..spin_count).for_each(|_| spin_loop());
                (start.elapsed().as_nanos() as u64 as f64) / (spin_count as f64)
            })
            .collect::<Vec<f64>>();
        attempts.into_iter().sum::<f64>() / (num_attempts as f64)
    };

    let print_work = |work_range: (u64, u64)| match work_range {
        (start, stop) if start == stop => format!("{:?}", Duration::from_nanos(start)),
        (start, stop) => format!(
            "{:?}-{:?}",
            Duration::from_nanos(start),
            Duration::from_nanos(stop)
        ),
    };

    for nanos in n_measure {
        for &(start, stop) in n_threads.iter() {
            for threads in start..=stop {
                for &work_locked in n_locked.iter() {
                    for &work_unlocked in n_unlocked.iter() {
                        println!(
                            "measure={:?} threads={} locked={} unlocked={}\n{}",
                            Duration::from_nanos(nanos.0),
                            threads,
                            print_work(work_locked),
                            print_work(work_unlocked),
                            "-".repeat(68),
                        );
                        print_results("", None);

                        bench_all(Bencher {
                            measure_ns: nanos.0,
                            num_threads: threads,
                            work_locked,
                            work_unlocked,
                            ns_per_work,
                        });
                    }
                }
            }
        }
    }
}

#[allow(unreachable_code)]
fn parse_arg<T: Copy + std::fmt::Debug, Args: Iterator<Item = String>>(
    args: &mut Args,
    arg_name: &'static str,
    supports_ranges: bool,
    mut decode: impl FnMut(&str) -> Option<T>,
) -> Vec<(T, T)> {
    let Some(arg) = args.next() else {
        print_help_and_abort(format!("Missing arg: {arg_name}"));
    };

    let mut parse_item = |part| match decode(part) {
        Some(value) => value,
        None => print_help_and_abort(format!("Invalid {arg_name}: {part}")),
    };

    arg.split(",")
        .map(|part| {
            let mut range = part.split("-");
            let first = parse_item(range.next().unwrap());
            match range.next() {
                Some(next) if supports_ranges => (first, parse_item(next)),
                Some(_) => print_help_and_abort(format!("{arg_name} doesnt support ranges")),
                None => (first, first),
            }
        })
        .collect()
}

fn print_help_and_abort(error_msg: String) -> ! {
    let exe = std::env::args().next().unwrap();
    unreachable!(
        r#"
{error_msg}
Usage: {exe} [measure] [threads] [locked] [unlocked]

where:
 [measure]:  [csv-ranged:time]  \\ List of time spent measuring for each mutex benchmark
 [threads]:  [csv-ranged:count] \\ List of thread counts for each benchmark
 [locked]:   [csv-ranged:time]  \\ List of time spent inside the lock for each benchmark
 [unlocked]: [csv-ranged:time]  \\ List of time spent outside the lock for each benchmark

where:
 [count]:     {{usize}}
 [time]:      {{u64}}[time_unit]
 [time_unit]: "ns" | "us" | "ms" | "s"

 [csv_ranged:{{rule}}]: 
      | {{rule}}                                          \\ single value
      | {{rule}} "-" {{rule}}                             \\ randomized value in range
      | [csv_ranged:{{rule}}] "," [csv_ranged:{{rule}}]   \\ multiple permutations
    "#
    );
}

#[derive(Copy, Clone, Debug)]
struct Bencher {
    measure_ns: u64,
    num_threads: usize,
    work_locked: (u64, u64),
    work_unlocked: (u64, u64),
    ns_per_work: f64,
}

impl Bencher {
    fn bench<CS: CriticalSection>(self) {
        let stopwatch = Arc::new((
            Barrier::new(self.num_threads + 1),
            AtomicBool::new(false),
            CS::new(),
        ));
        let threads = (0..self.num_threads)
            .map(|i| {
                let stopwatch = stopwatch.clone();
                thread::spawn(move || {
                    let mut prng = (i + 1) as u32;
                    let mut gen_work = |(lo, hi)| {
                        let ns = if lo == hi {
                            lo
                        } else {
                            prng = prng.wrapping_mul(134775813).wrapping_add(1);
                            ((prng as u64) % (hi - lo + 1)) + lo
                        };
                        (self.ns_per_work * (ns as f64)) as u64
                    };

                    stopwatch.0.wait();
                    (0u64..)
                        .take_while(|_| !stopwatch.1.load(Ordering::Acquire))
                        .map(|n| {
                            let work_in = gen_work(self.work_locked);
                            let work_out = gen_work(self.work_unlocked);
                            stopwatch.2.with(|| (0..work_in).for_each(|_| spin_loop()));
                            (0..work_out).for_each(|_| spin_loop());
                            n
                        })
                        .last()
                        .unwrap_or(0)
                })
            })
            .collect::<Vec<_>>();

        stopwatch.0.wait();
        thread::sleep(Duration::from_nanos(self.measure_ns));
        stopwatch.1.store(true, Ordering::Release);

        print_results(
            CS::NAME,
            Some(threads.into_iter().map(|t| t.join().unwrap()).collect()),
        );
    }
}

fn print_results(name: &'static str, results: Option<Vec<u64>>) {
    let lower = |value: f64| {
        if value < 1_000f64 {
            format!("{}", value.round())
        } else if value < 1_000_000f64 {
            format!("{}k", (value / 1_000f64).round())
        } else if value < 1_000_000_000f64 {
            format!("{:.2}m", value / 1_000_000f64)
        } else {
            format!("{:.2}b", value / 1_000_000_000f64)
        }
    };

    let (name, sum, mean, min, max, stdev) = match results {
        None => (
            "name".to_string(),
            "sum".to_string(),
            "mean".to_string(),
            "min".to_string(),
            "max".to_string(),
            "stdev".to_string(),
        ),
        Some(mut results) => {
            let sum = results.iter().sum::<u64>() as f64;
            let mean = sum / (results.len() as f64);
            let mut stdev = results.iter().fold(0f64, |stdev, &n| {
                let r = (n as f64) - mean;
                stdev + (r * r)
            });
            if results.len() > 1 {
                stdev = (stdev / ((results.len() - 1) as f64)).sqrt();
            }

            results.sort();
            let min = results[0] as f64;
            let max = results.last().copied().unwrap() as f64;
            (
                name.to_string(),
                lower(sum),
                lower(mean),
                lower(min),
                lower(max),
                lower(stdev),
            )
        }
    };

    println!(
        "{:<18} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7}",
        name, sum, mean, min, max, stdev
    )
}
