use crossbeam_utils::Backoff;
use std::{
    cell::UnsafeCell,
    mem::{ManuallyDrop, MaybeUninit},
    pin::pin,
    ptr::{NonNull, dangling_mut, null_mut},
    sync::atomic::{AtomicPtr, Ordering, fence},
    thread::{self, Thread},
};

#[derive(Default)]
struct Parker(AtomicPtr<()>);
impl Parker {
    fn park(&self) {
        let b = Backoff::new();
        while self.0.load(Ordering::Relaxed).is_null() {
            b.spin();
            if b.is_completed() {
                self.park_slow();
                break;
            }
        }
        fence(Ordering::Acquire);
    }

    #[cold]
    fn park_slow(&self) {
        let event = pin!((
            UnsafeCell::new(ManuallyDrop::new(thread::current())),
            AtomicPtr::<()>::new(null_mut()),
        ));
        let ptr = self.0.swap(
            &*event as *const _ as *const () as *mut (),
            Ordering::Release,
        );
        if ptr.is_null() {
            while event.1.load(Ordering::Relaxed).is_null() {
                thread::park();
            }
        } else {
            unsafe { ManuallyDrop::drop(&mut *event.0.get()) };
        }
    }

    fn unpark(&self) {
        let ptr = self.0.swap(dangling_mut(), Ordering::Release);
        if !ptr.is_null() {
            fence(Ordering::Acquire);
            let event = ptr as *mut (UnsafeCell<ManuallyDrop<Thread>>, AtomicPtr<()>);
            unsafe {
                let thread = ManuallyDrop::take(&mut *(*event).0.get());
                (*event).1.store(dangling_mut(), Ordering::Release);
                thread.unpark();
            }
        }
    }
}

pub struct Node {
    next: AtomicPtr<Self>,
    callback: fn(*mut Node),
    parker: Parker,
}

pub trait Queue: Send + Sync + Default {
    fn submit(&self, node: *mut Node);
}

#[derive(Default)]
pub struct Mutex<Q: Queue> {
    queue: Q,
}

impl<Q: Queue> Mutex<Q> {
    pub fn with<F: FnOnce()>(&self, f: F) {
        struct Submitter<F: FnOnce()> {
            func: ManuallyDrop<F>,
            node: Node,
        }
        impl<F: FnOnce()> Submitter<F> {
            fn callback(node: *mut Node) {
                unsafe {
                    let base = MaybeUninit::<Self>::uninit();
                    let node_ptr = (&raw const (*base.as_ptr()).node) as *const u8;
                    let node_offset = node_ptr.byte_offset_from(base.as_ptr() as *const u8);
                    let this_ptr = (node as *mut u8).byte_sub(node_offset as usize) as *mut Self;
                    ManuallyDrop::take(&mut (*this_ptr).func)();
                }
            }
        }
        let submitter = pin!(Submitter {
            func: ManuallyDrop::new(f),
            node: Node {
                next: AtomicPtr::new(null_mut()),
                callback: Submitter::<F>::callback,
                parker: Parker::default(),
            },
        });
        self.queue.submit((&raw const submitter.node) as *mut Node);
        submitter.node.parker.park();
    }
}

#[derive(Default)]
pub struct LockFree {
    top: AtomicPtr<Node>,
}

impl Queue for LockFree {
    fn submit(&self, node: *mut Node) {
        let b = Backoff::new();
        loop {
            let top = self.top.load(Ordering::Relaxed);
            unsafe { (*node).next.store(top, Ordering::Relaxed) };
            match self
                .top
                .compare_exchange(top, node, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) if top.is_null() => return self.process(node),
                Ok(_) => return,
                Err(_) => b.spin(),
            }
        }
    }
}

impl LockFree {
    #[cold]
    fn process(&self, mut top: *mut Node) {
        unsafe {
            let mut bottom = null_mut();
            let mut unparked = null_mut();
            loop {
                fence(Ordering::Acquire);

                let mut node = top;
                loop {
                    let next = (*node).next.load(Ordering::Relaxed);
                    (*node).next.store(unparked, Ordering::Relaxed);
                    unparked = node;

                    ((*node).callback)(node);
                    node = next;
                    if node == bottom {
                        break;
                    }
                }

                if let Err(new_top) =
                    self.top
                        .compare_exchange(top, null_mut(), Ordering::Release, Ordering::Relaxed)
                {
                    bottom = top;
                    top = new_top;
                    continue;
                }

                return while let Some(node) = NonNull::new(unparked) {
                    unparked = node.as_ref().next.load(Ordering::Relaxed);
                    node.as_ref().parker.unpark();
                };
            }
        }
    }
}

#[derive(Default)]
pub struct WaitFree {
    tail: AtomicPtr<Node>,
}

impl Queue for WaitFree {
    fn submit(&self, node: *mut Node) {
        unsafe {
            (*node).next.store(node, Ordering::Relaxed);

            let tail = self.tail.swap(node, Ordering::Release);
            if tail.is_null() {
                return self.process(null_mut(), node);
            }

            fence(Ordering::Acquire);
            if (*tail)
                .next
                .compare_exchange(tail, node, Ordering::Release, Ordering::Acquire)
                .is_err()
            {
                return self.process(tail, node);
            }
        }
    }
}

impl WaitFree {
    #[cold]
    fn process(&self, mut unparked: *mut Node, mut node: *mut Node) {
        unsafe {
            loop {
                ((*node).callback)(node);

                let mut next = (*node).next.load(Ordering::Acquire);
                if next == node {
                    if self
                        .tail
                        .compare_exchange(node, null_mut(), Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        (*node).parker.unpark();
                        return while let Some(node) = NonNull::new(unparked) {
                            unparked = node.as_ref().next.load(Ordering::Relaxed);
                            node.as_ref().parker.unpark();
                        };
                    }

                    match (*node).next.compare_exchange(
                        node,
                        unparked,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return,
                        Err(new) => next = new,
                    }
                }

                (*node).next.store(unparked, Ordering::Relaxed);
                unparked = node;
                node = next;
            }
        }
    }
}
