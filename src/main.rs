use arbitrary::{Error, Unstructured};
use rand::RngCore;
use rusty_v8 as v8;
use std::convert::TryFrom;
use wasmtime::*;

const N: usize = 8;

fn main() {
    env_logger::init();

    let platform = v8::new_default_platform(0, false).make_shared();
    v8::V8::initialize_platform(platform);
    v8::V8::initialize();

    let threads = (0..N)
        .map(|_| {
            std::thread::spawn(|| {
                let mut rng = rand::thread_rng();
                let mut data = Vec::new();
                let mut isolate = v8::Isolate::new(Default::default());
                loop {
                    data.truncate(0);
                    data.resize(1024, 0);
                    rng.fill_bytes(&mut data);
                    loop {
                        match run_once(&mut data, &mut isolate) {
                            Ok(()) => break,
                            Err(Error::NotEnoughData) => {
                                let cur = data.len();
                                let extra = 1024;
                                data.resize(cur + extra, 0);
                                rng.fill_bytes(&mut data[cur..]);
                            }
                            Err(e) => panic!("failed to generated module: {}", e),
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    for thread in threads {
        thread.join().unwrap();
    }
}

fn run_once(data: &[u8], isolate: &mut v8::Isolate) -> Result<(), Error> {
    let mut u = Unstructured::new(data);
    let mut wasm = wasm_smith::Module::new(SingleFunctionModuleConfig, &mut u)?;
    wasm.ensure_termination(1_000);
    let wasm = wasm.to_bytes();
    log::trace!("generated wasm");
    std::fs::write("input.wasm", &wasm).unwrap();

    // Wasmtime setup
    let mut config = Config::new();
    config.wasm_simd(true);
    let wasmtime_engine = Engine::new(&config).unwrap();
    let mut wasmtime_store = wasmtime::Store::new(&wasmtime_engine, ());

    // V8 setup
    let mut scope = v8::HandleScope::new(isolate);
    let context = v8::Context::new(&mut scope);
    let global = context.global(&mut scope);
    let mut scope = v8::ContextScope::new(&mut scope, context);

    // Wasmtime: compile module
    let wasmtime_module = wasmtime::Module::new(&wasmtime_engine, &wasm).unwrap();
    log::trace!("compiled module with wasmtime");

    // V8: compile module
    let buf = v8::ArrayBuffer::new_backing_store_from_boxed_slice(wasm.into());
    let buf = v8::SharedRef::from(buf);
    let name = v8::String::new(&mut scope, "WASM_BINARY").unwrap();
    let buf = v8::ArrayBuffer::with_backing_store(&mut scope, &buf);
    global.set(&mut scope, name.into(), buf.into());
    let v8_module = eval(&mut scope, "new WebAssembly.Module(WASM_BINARY)").unwrap();
    let name = v8::String::new(&mut scope, "WASM_MODULE").unwrap();
    global.set(&mut scope, name.into(), v8_module);
    log::trace!("compiled module with v8");

    // Wasmtime: instantiate
    let wasmtime_instance = wasmtime::Instance::new(&mut wasmtime_store, &wasmtime_module, &[]);

    // V8: instantiate
    let v8_instance = eval(&mut scope, "new WebAssembly.Instance(WASM_MODULE)");

    // Verify V8 and wasmtime match
    let (wasmtime_instance, v8_instance) = match (wasmtime_instance, v8_instance) {
        (Ok(i1), Ok(i2)) => (i1, i2),
        (Ok(_), Err(msg)) => {
            panic!("wasmtime succeeded at instantiation, v8 failed: {}", msg)
        }
        (Err(err), Ok(_)) => {
            panic!("v8 succeeded at instantiation, wasmtime failed: {:?}", err)
        }
        (Err(err), Err(msg)) => {
            assert_error_matches(&err, &msg);
            return Ok(());
        }
    };

    let (func, ty) = match first_exported_function(&wasmtime_module) {
        Some(f) => f,
        None => return Ok(()),
    };

    // not supported yet in V8
    if ty.params().chain(ty.results()).any(|t| t == ValType::V128) {
        return Ok(());
    }

    let mut wasmtime_params = Vec::new();
    let mut v8_params = Vec::new();
    for param in ty.params() {
        wasmtime_params.push(match param {
            ValType::I32 => Val::I32(0),
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0),
            ValType::F64 => Val::F64(0),
            _ => unimplemented!(),
        });
        v8_params.push(match param {
            ValType::I32 | ValType::F32 | ValType::F64 => v8::Number::new(&mut scope, 0.0).into(),
            ValType::I64 => v8::BigInt::new_from_i64(&mut scope, 0).into(),
            _ => unimplemented!(),
        });
    }

    // Wasmtime: call the first exported func
    let wasmtime_main = wasmtime_instance
        .get_func(&mut wasmtime_store, func)
        .expect("function export is present");
    let wasmtime_vals = wasmtime_main.call(&mut wasmtime_store, &wasmtime_params);

    // V8: call the first exported func
    let name = v8::String::new(&mut scope, "WASM_INSTANCE").unwrap();
    global.set(&mut scope, name.into(), v8_instance);
    let name = v8::String::new(&mut scope, "EXPORT_NAME").unwrap();
    let func_name = v8::String::new(&mut scope, func).unwrap();
    global.set(&mut scope, name.into(), func_name.into());
    let name = v8::String::new(&mut scope, "ARGS").unwrap();
    let v8_params = v8::Array::new_with_elements(&mut scope, &v8_params);
    global.set(&mut scope, name.into(), v8_params.into());
    let v8_vals = eval(
        &mut scope,
        &format!("WASM_INSTANCE.exports[EXPORT_NAME](...ARGS)"),
    );

    // Verify V8 and wasmtime match
    match (wasmtime_vals, v8_vals) {
        (Ok(wasmtime), Ok(v8)) => {
            match wasmtime.len() {
                0 => assert!(v8.is_undefined()),
                1 => assert_val_match(&wasmtime[0], &v8, &mut scope),
                _ => {
                    let array = v8::Local::<'_, v8::Array>::try_from(v8).unwrap();
                    for (i, wasmtime) in wasmtime.iter().enumerate() {
                        let v8 = array.get_index(&mut scope, i as u32).unwrap();
                        assert_val_match(wasmtime, &v8, &mut scope);
                        // ..
                    }
                }
            }
        }
        (Ok(_), Err(msg)) => {
            panic!("wasmtime succeeded at invocation, v8 failed: {}", msg)
        }
        (Err(err), Ok(_)) => {
            panic!("v8 succeeded at invocation, wasmtime failed: {:?}", err)
        }
        (Err(err), Err(msg)) => {
            assert_error_matches(&err, &msg);
            return Ok(());
        }
    };

    // Verify V8 and wasmtime match memories
    if let Some(mem) = first_exported_memory(&wasmtime_module) {
        let wasmtime = wasmtime_instance
            .get_memory(&mut wasmtime_store, mem)
            .unwrap();

        let name = v8::String::new(&mut scope, "MEMORY_NAME").unwrap();
        let func_name = v8::String::new(&mut scope, mem).unwrap();
        global.set(&mut scope, name.into(), func_name.into());
        let v8 = eval(
            &mut scope,
            &format!("WASM_INSTANCE.exports[MEMORY_NAME].buffer"),
        )
        .unwrap();
        let v8 = v8::Local::<'_, v8::ArrayBuffer>::try_from(v8).unwrap();
        let v8_data = v8.get_backing_store();
        let wasmtime_data = wasmtime.data(&wasmtime_store);
        assert_eq!(wasmtime_data.len(), v8_data.len());
        for i in 0..v8_data.len() {
            if wasmtime_data[i] != v8_data[i].get() {
                panic!("memories differ");
            }
        }
    }

    return Ok(());

    fn eval<'s>(
        scope: &mut v8::HandleScope<'s>,
        code: &str,
    ) -> Result<v8::Local<'s, v8::Value>, String> {
        let mut tc = v8::TryCatch::new(scope);
        let mut scope = v8::EscapableHandleScope::new(&mut tc);
        let source = v8::String::new(&mut scope, code).unwrap();
        let script = v8::Script::compile(&mut scope, source, None).unwrap();
        match script.run(&mut scope) {
            Some(val) => Ok(scope.escape(val)),
            None => {
                drop(scope);
                assert!(tc.has_caught());
                Err(tc
                    .message()
                    .unwrap()
                    .get(&mut tc)
                    .to_rust_string_lossy(&mut tc))
            }
        }
    }

    fn assert_val_match(a: &Val, b: &v8::Local<'_, v8::Value>, scope: &mut v8::HandleScope<'_>) {
        match *a {
            Val::I32(wasmtime) => {
                assert_eq!(i64::from(wasmtime), b.to_int32(scope).unwrap().value());
            }
            Val::I64(wasmtime) => {
                assert_eq!((wasmtime, true), b.to_big_int(scope).unwrap().i64_value());
            }
            Val::F32(wasmtime) => {
                same_float(
                    f64::from(f32::from_bits(wasmtime)),
                    b.to_number(scope).unwrap().value(),
                );
            }
            Val::F64(wasmtime) => {
                same_float(
                    f64::from_bits(wasmtime),
                    b.to_number(scope).unwrap().value(),
                );
            }
            _ => panic!("unsupported match {:?}", a),
        }

        fn same_float(a: f64, b: f64) {
            assert!(a == b || (a.is_nan() && b.is_nan()), "{} != {}", a, b);
        }
    }

    fn assert_error_matches(wasmtime: &anyhow::Error, v8: &str) {
        let wasmtime_msg = match wasmtime.downcast_ref::<Trap>() {
            Some(trap) => trap.display_reason().to_string(),
            None => format!("{:?}", wasmtime),
        };
        let verify_wasmtime = |msg: &str| {
            assert!(wasmtime_msg.contains(msg), "{}\n!=\n{}", wasmtime_msg, v8);
        };
        let verify_v8 = |msg: &[&str]| {
            assert!(
                msg.iter().any(|msg| v8.contains(msg)),
                "{:?}\n\t!=\n{}",
                wasmtime_msg,
                v8
            );
        };
        if let Some(code) = wasmtime.downcast_ref::<Trap>().and_then(|t| t.trap_code()) {
            match code {
                TrapCode::MemoryOutOfBounds => {
                    return verify_v8(&[
                        "memory access out of bounds",
                        "data segment is out of bounds",
                    ])
                }
                TrapCode::UnreachableCodeReached => return verify_v8(&["unreachable"]),
                TrapCode::IntegerDivisionByZero => {
                    return verify_v8(&["divide by zero", "remainder by zero"])
                }
                TrapCode::StackOverflow => return verify_v8(&["call stack size exceeded"]),
                TrapCode::IndirectCallToNull => return verify_v8(&["null function"]),
                TrapCode::TableOutOfBounds => {
                    return verify_v8(&[
                        "table initializer is out of bounds",
                        "table index is out of bounds",
                    ])
                }
                TrapCode::BadSignature => return verify_v8(&["function signature mismatch"]),
                TrapCode::IntegerOverflow | TrapCode::BadConversionToInteger => {
                    return verify_v8(&[
                        "float unrepresentable in integer range",
                        "divide result unrepresentable",
                    ])
                }
                other => log::debug!("unknown code {:?}", other),
            }
        }
        verify_wasmtime("xxxxxxxxxxxxxxxxxxxxxxxx");
    }

    fn first_exported_function(module: &Module) -> Option<(&str, FuncType)> {
        for e in module.exports() {
            match e.ty() {
                ExternType::Func(t) => return Some((e.name(), t)),
                _ => {}
            }
        }
        None
    }

    fn first_exported_memory(module: &Module) -> Option<&str> {
        for e in module.exports() {
            match e.ty() {
                ExternType::Memory(..) => return Some(e.name()),
                _ => {}
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct SingleFunctionModuleConfig;

impl wasm_smith::Config for SingleFunctionModuleConfig {
    fn allow_start_export(&self) -> bool {
        false
    }

    fn min_types(&self) -> usize {
        1
    }

    fn min_funcs(&self) -> usize {
        1
    }

    fn max_funcs(&self) -> usize {
        1
    }

    fn min_memories(&self) -> u32 {
        1
    }

    fn max_memories(&self) -> usize {
        1
    }

    fn max_imports(&self) -> usize {
        0
    }

    fn min_exports(&self) -> usize {
        2
    }

    fn max_memory_pages(&self, _is_64: bool) -> u64 {
        1
    }

    fn memory_max_size_required(&self) -> bool {
        true
    }

    fn bulk_memory_enabled(&self) -> bool {
        true
    }
    fn simd_enabled(&self) -> bool {
        true
    }

    // NaN is canonicalized at the wasm level for differential fuzzing so we
    // can paper over NaN differences between engines.
    fn canonicalize_nans(&self) -> bool {
        true
    }
}
