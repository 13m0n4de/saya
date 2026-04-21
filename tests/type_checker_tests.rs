use saya::hir::*;
use saya::lexer::Lexer;
use saya::parser::Parser;
use saya::type_checker::TypeChecker;
use saya::types::{TypeContext, TypeId, TypeKind};

macro_rules! typecheck {
    ($input:expr) => {{
        let lexer = Lexer::new($input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();
        let mut type_context = TypeContext::new();
        let mut type_checker =
            TypeChecker::new(&mut type_context, None, std::collections::HashMap::new());
        type_checker.check(&program)
    }};
}

#[test]
fn test_integer_literal() {
    let program = typecheck!("fn main() -> u8 { 255u8 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            assert_eq!(func.body.as_ref().unwrap().type_id, TypeId::U8);
        }
        _ => panic!("Expected function"),
    }

    let program = typecheck!("fn main() -> u16 { 1000u16 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            assert_eq!(func.body.as_ref().unwrap().type_id, TypeId::U16);
        }
        _ => panic!("Expected function"),
    }

    let program = typecheck!("fn main() -> u32 { 70000u32 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            assert_eq!(func.body.as_ref().unwrap().type_id, TypeId::U32);
        }
        _ => panic!("Expected function"),
    }

    let program = typecheck!("fn main() -> i32 { 100i32 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            assert_eq!(func.body.as_ref().unwrap().type_id, TypeId::I32);
        }
        _ => panic!("Expected function"),
    }

    let program = typecheck!("fn main() -> i64 { 42 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            assert_eq!(body.type_id, TypeId::I64);
        }
        _ => panic!("Expected function"),
    }

    // out of range
    assert!(typecheck!("fn main() -> u8 { 256u8 }").is_err());
    assert!(typecheck!("fn main() -> u16 { 65536u16 }").is_err());
    assert!(typecheck!("fn main() -> u32 { 4294967296u32 }").is_err());
    assert!(typecheck!("fn main() -> i32 { 2147483648i32 }").is_err());
    assert!(typecheck!("fn main() -> i32 { -2147483649i32 }").is_err());
}

#[test]
fn test_float_literal() {
    let program = typecheck!("fn main() -> f32 { 1.5f32 }").unwrap();
    match &program.items[0].kind {
        ItemKind::Function(func) => {
            assert_eq!(func.body.as_ref().unwrap().type_id, TypeId::F32);
        }
        _ => panic!("Expected function"),
    }

    let program = typecheck!("fn main() -> f64 { 3.14 }").unwrap();

    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            assert_eq!(body.type_id, TypeId::F64);
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_string_literal() {
    let code = r#"fn main() -> [u8] { "hello" }"#;
    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer).unwrap();
    let program = parser.parse().unwrap();
    let mut type_context = TypeContext::new();
    let mut type_checker =
        TypeChecker::new(&mut type_context, None, std::collections::HashMap::new());
    let program = type_checker.check(&program).unwrap();

    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            let ty = type_context.get(body.type_id);
            assert!(matches!(ty.kind, TypeKind::Slice(TypeId::U8)));
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_cstring_literal() {
    let code = r#"fn main() -> *u8 { c"hello C FFI" }"#;
    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer).unwrap();
    let program = parser.parse().unwrap();
    let mut type_context = TypeContext::new();
    let mut type_checker =
        TypeChecker::new(&mut type_context, None, std::collections::HashMap::new());
    let program = type_checker.check(&program).unwrap();

    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            let ty = type_context.get(body.type_id);
            assert!(matches!(ty.kind, TypeKind::Pointer(TypeId::U8)));
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_bool_literal() {
    let program = typecheck!("fn main() -> bool { true }").unwrap();

    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            assert_eq!(body.type_id, TypeId::Bool);
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_simple_let_binding() {
    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; x }");
    assert!(result.is_ok());
}

#[test]
fn test_arithmetic_operations() {
    let result = typecheck!("fn test() -> i64 { 1 + 2 * 3 - 4 / 2 }");
    assert!(result.is_ok());
}

#[test]
fn test_comparison_operations() {
    let result = typecheck!("fn test() -> bool { 1 < 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { 1 <= 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { 1 > 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { 1 >= 2 }");
    assert!(result.is_ok());
}

#[test]
fn test_equality_operations() {
    let result = typecheck!("fn test() -> bool { 1 == 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { 1 != 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { true == false }");
    assert!(result.is_ok());
}

#[test]
fn test_logical_operations() {
    let result = typecheck!("fn test() -> bool { true && false }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { true || false }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { !true }");
    assert!(result.is_ok());
}

#[test]
fn test_unary_operators() {
    let result = typecheck!("fn test() -> i64 { -42 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> bool { !true }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { !42 }");
    assert!(result.is_ok());
}

#[test]
fn test_if_else_expression() {
    let result = typecheck!("fn test() -> i64 { if true { 1 } else { 2 } }");
    assert!(result.is_ok());
}

#[test]
fn test_if_without_else() {
    let result = typecheck!("fn test() { if true { 1; } }");
    assert!(result.is_ok());
}

#[test]
fn test_while_loop() {
    let result = typecheck!("fn test() { while true { break; } }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() { loop { break; } }");
    assert!(result.is_ok());
}

#[test]
fn test_loop_break_value() {
    let result = typecheck!("fn test() -> i64 { loop {} }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() { loop { break; } }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { loop { break 42; } }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { loop { if true { break 1; } break 2; } }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { let x: i64 = loop { break 1; }; x }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() { loop { if true { break 1; } else { break true; } } }");
    assert!(result.is_err());

    let result = typecheck!("fn test() -> i64 { loop { break true; } }");
    assert!(result.is_err());
}

#[test]
fn test_break_continue() {
    let result = typecheck!("fn test() { while true { if true { break; } else { continue; } } }");
    assert!(result.is_ok());
}

#[test]
fn test_return_statement() {
    let result = typecheck!("fn test() -> i64 { return 42; }");
    assert!(result.is_ok());
}

#[test]
fn test_return_without_value() {
    let result = typecheck!("fn test() { return; }");
    assert!(result.is_ok());
}

#[test]
fn test_array_literal() {
    let result = typecheck!("fn test() -> [i64; 3] { [1, 2, 3] }");
    assert!(result.is_ok());
}

#[test]
fn test_array_repeat() {
    let result = typecheck!("fn test() -> [i64; 5] { [42; 5] }");
    assert!(result.is_ok());
}

#[test]
fn test_array_index() {
    let result = typecheck!("fn test() -> i64 { let arr: [i64; 3] = [1, 2, 3]; arr[0] }");
    assert!(result.is_ok());

    let result = typecheck!(
        "fn test() -> i64 { let arr: [i64; 3] = [1, 2, 3]; let ptr: *[i64; 3] = &arr; ptr[0] }"
    );
    assert!(result.is_ok());
}

#[test]
fn test_function_call() {
    let result = typecheck!(
        r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 { add(1, 2) }
        "#
    );
    assert!(result.is_ok());
}

#[test]
fn test_function_call_no_args() {
    let result = typecheck!(
        r#"
        fn get_value() -> i64 { 42 }
        fn main() -> i64 { get_value() }
        "#
    );
    assert!(result.is_ok());
}

#[test]
fn test_assignment() {
    let result = typecheck!("fn test() { let x: i64 = 1; x = 2; }");
    assert!(result.is_ok());
}

#[test]
fn test_const_definition() {
    let result = typecheck!("const PI: i64 = 3;");
    assert!(result.is_ok());
}

#[test]
fn test_static_definition() {
    let result = typecheck!("static GLOBAL: i64 = 42;");
    assert!(result.is_ok());
}

#[test]
fn test_access_global() {
    let result = typecheck!(
        r#"
        const X: i64 = 10;
        fn test() -> i64 { X }
        "#
    );
    assert!(result.is_ok());
}

#[test]
fn test_block_expression() {
    let result = typecheck!("fn test() -> i64 { { let x: i64 = 1; x + 1 } }");
    assert!(result.is_ok());
}

#[test]
fn test_nested_scopes() {
    let result = typecheck!(
        r#"
        fn test() -> i64 {
            let x: i64 = 1;
            {
                let y: i64 = 2;
                x + y
            }
        }
        "#
    );
    assert!(result.is_ok());
}

#[test]
fn test_external_function() {
    let result = typecheck!("extern fn external(x: i64) -> i64;");
    assert!(result.is_ok());
}

#[test]
fn test_variadic_function() {
    let result = typecheck!("extern fn printf(fmt: *u8, ...) -> i64;");
    assert!(result.is_ok());

    let result = typecheck!(
        r#"
        extern fn printf(fmt: *u8, ...) -> i64;
        fn main() -> i64 { printf(c"hello") }
        "#
    );
    assert!(result.is_ok());

    let program = typecheck!(
        r#"
        extern fn printf(fmt: *u8, ...) -> i64;
        fn main() -> i64 { printf(c"%d", 42) }
        "#
    )
    .unwrap();
    let ItemKind::Function(func) = &program.items[1].kind else {
        panic!("expected function");
    };
    let body = func.body.as_ref().unwrap();
    let StmtKind::Expr(Expr {
        kind: ExprKind::Call(call),
        ..
    }) = &body.stmts[0].kind
    else {
        panic!("expected call");
    };
    assert_eq!(call.variadic_start, Some(1));

    let result = typecheck!(
        r#"
        extern fn printf(fmt: *u8, ...) -> i64;
        fn main() -> i64 { printf() }
        "#
    );
    assert!(result.is_err());
}

#[test]
fn test_bitwise_operators() {
    let result = typecheck!("fn test() -> i64 { 1 & 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { 1 | 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> u16 { 0u16 & 255u16 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> u32 { 0u32 | 1u32 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i32 { 0i32 & 1i32 }");
    assert!(result.is_ok());

    // not integer
    let result = typecheck!("fn test() -> f32 { 1.0f32 & 2.0f32 }");
    assert!(result.is_err());
}

#[test]
fn test_type_mismatch_let_binding() {
    let result = typecheck!("fn test() { let x: i64 = true; }");
    assert!(result.is_err());
}

#[test]
fn test_undefined_variable() {
    let result = typecheck!("fn test() -> i64 { x }");
    assert!(result.is_err());
}

#[test]
fn test_function_return_type_mismatch() {
    let result = typecheck!("fn test() -> i64 { true }");
    assert!(result.is_err());
}

#[test]
fn test_undefined_function() {
    let result = typecheck!("fn test() -> i64 { foo() }");
    assert!(result.is_err());
}

#[test]
fn test_function_arg_count_mismatch() {
    let result = typecheck!(
        r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 { add(1) }
        "#
    );
    assert!(result.is_err());
}

#[test]
fn test_function_arg_type_mismatch() {
    let result = typecheck!(
        r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 { add(1, true) }
        "#
    );
    assert!(result.is_err());
}

#[test]
fn test_arithmetic_type_error() {
    let result = typecheck!("fn test() -> i64 { 1 + true }");
    assert!(result.is_err());
}

#[test]
fn test_comparison_type_error() {
    let result = typecheck!("fn test() -> bool { 1 < true }");
    assert!(result.is_err());
}

#[test]
fn test_logical_operator_type_error() {
    let result = typecheck!("fn test() -> bool { 1 && 2 }");
    assert!(result.is_err());
}

#[test]
fn test_equality_type_error() {
    let result = typecheck!("fn test() -> bool { 1 == true }");
    assert!(result.is_err());
}

#[test]
fn test_unary_neg_type_error() {
    let result = typecheck!("fn test() -> i64 { -true }");
    assert!(result.is_err());
}

#[test]
fn test_unary_not_type_error() {
    let result = typecheck!(r#"fn test() -> bool { !"hello" }"#);
    assert!(result.is_err());
}

#[test]
fn test_if_condition_not_bool() {
    let result = typecheck!("fn test() -> i64 { if 1 { 2 } else { 3 } }");
    assert!(result.is_err());
}

#[test]
fn test_while_condition_not_bool() {
    let result = typecheck!("fn test() { while 1 { break; } }");
    assert!(result.is_err());
}

#[test]
fn test_if_else_branch_type_mismatch() {
    let result = typecheck!("fn test() -> i64 { if true { 1 } else { true } }");
    assert!(result.is_err());
}

#[test]
fn test_assignment_type_mismatch() {
    let result = typecheck!("fn test() { let x: i64 = 1; x = true; }");
    assert!(result.is_err());
}

#[test]
fn test_return_type_mismatch() {
    let result = typecheck!("fn test() -> i64 { return true; }");
    assert!(result.is_err());
}

#[test]
fn test_return_missing_value() {
    let result = typecheck!("fn test() -> i64 { return; }");
    assert!(result.is_err());
}

#[test]
fn test_array_element_type_mismatch() {
    let result = typecheck!("fn test() -> [i64; 3] { [1, 2, true] }");
    assert!(result.is_err());
}

#[test]
fn test_empty_array_inference() {
    let result = typecheck!("fn test() { let x: [i64; 0] = []; }");
    assert!(result.is_err());
}

#[test]
fn test_array_index_not_i64() {
    let result = typecheck!("fn test() -> i64 { let arr: [i64; 3] = [1, 2, 3]; arr[true] }");
    assert!(result.is_err());
}

#[test]
fn test_index_non_array() {
    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; x[0] }");
    assert!(result.is_err());

    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; let ptr: *i64 = &x; ptr[0] }");
    assert!(result.is_err());
}

#[test]
fn test_repeat_count_not_i64() {
    let result = typecheck!("fn test() -> [i64; 5] { [42; true] }");
    assert!(result.is_err());
}

#[test]
fn test_repeat_count_not_constant() {
    let result = typecheck!("fn test() { let n: i64 = 5; let arr: [i64; 5] = [42; n]; }");
    assert!(result.is_err());
}

#[test]
fn test_const_type_mismatch() {
    let result = typecheck!("const X: i64 = true;");
    assert!(result.is_err());
}

#[test]
fn test_static_type_mismatch() {
    let result = typecheck!("static X: i64 = true;");
    assert!(result.is_err());
}

#[test]
fn test_unreachable_after_return() {
    let result = typecheck!("fn test() -> i64 { return 1; 2 }");
    assert!(result.is_err());
}

#[test]
fn test_never_type_compatibility() {
    let result = typecheck!("fn test() -> i64 { return 42; }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { if true { return 1; } else { return 2; } }");
    assert!(result.is_ok());
}

#[test]
fn test_if_else_with_never() {
    let result = typecheck!("fn test() -> i64 { if true { return 1; } else { 2 } }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { if true { 1 } else { return 2; } }");
    assert!(result.is_ok());
}

#[test]
fn test_pointer_basic() {
    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; let ptr: *i64 = &x; *ptr }");
    assert!(result.is_ok());
}

#[test]
fn test_pointer_double() {
    let result = typecheck!(
        "fn test() -> i64 { let x: i64 = 42; let ptr: *i64 = &x; let pptr: **i64 = &ptr; **pptr }"
    );
    assert!(result.is_ok());
}

#[test]
fn test_pointer_to_global() {
    let result = typecheck!(
        r#"
        static GLOBAL: i64 = 100;
        fn test() -> i64 { let ptr: *i64 = &GLOBAL; *ptr }
        "#
    );
    assert!(result.is_ok());
}

#[test]
fn test_pointer_type_error() {
    let result = typecheck!("fn test() { let x: i64 = 42; let ptr: *bool = &x; }");
    assert!(result.is_err());
}

#[test]
fn test_deref_non_pointer() {
    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; *x }");
    assert!(result.is_err());
}

#[test]
fn test_ref_constant() {
    let result = typecheck!(
        r#"
        const X: i64 = 42;
        fn test() -> *i64 { &X }
        "#
    );
    assert!(result.is_err());
}

#[test]
fn test_structs() {
    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() -> i64 { let p: Point = Point { x: 1, y: 2 }; p.x }
          "#
    );
    assert!(result.is_ok());

    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() -> i64 { let p: Point = Point { x: 10, y: 20 }; let ptr: *Point = &p; ptr.x }
          "#
    );
    assert!(result.is_ok());

    let result = typecheck!(
        r#"
          struct Node { value: i64, next: *Node }
          "#
    );
    assert!(result.is_ok());

    let result = typecheck!(
        r#"
          struct A { b_ptr: *B }
          struct B { a_ptr: *A, value: i64 }
          "#
    );
    assert!(result.is_ok());

    let result = typecheck!(
        r#"
          struct A { b: B }
          struct B { a: A }
          "#
    );
    assert!(result.is_err());

    let result = typecheck!("struct C { c: C }");
    assert!(result.is_err());

    let result = typecheck!("struct Point { x: i64, x: i64 }");
    assert!(result.is_err());

    let result = typecheck!("struct Point { x: Foo }");
    assert!(result.is_err());

    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() { let p: Point = Point { x: 1 }; }
          "#
    );
    assert!(result.is_err());

    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() { let p: Point = Point { x: 1, y: true }; }
          "#
    );
    assert!(result.is_err());

    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() { let p: Point = Point { x: 1, y: 2, z: 3 }; }
          "#
    );
    assert!(result.is_err());

    let result = typecheck!(
        r#"
          struct Point { x: i64, y: i64 }
          fn test() -> i64 { let p: Point = Point { x: 1, y: 2 }; p.z }
          "#
    );
    assert!(result.is_err());

    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; x.field }");
    assert!(result.is_err());

    let result = typecheck!("fn test() -> i64 { let x: i64 = 42; let ptr: *i64 = &x; ptr.field }");
    assert!(result.is_err());
}

#[test]
fn test_type_alias() {
    assert!(typecheck!("type MyInt = i64; fn test() -> MyInt { let x: MyInt = 42; x }").is_ok());

    let program = typecheck!("type MyInt = i64; fn test() -> MyInt { 42 }").unwrap();
    let ItemKind::Function(func) = &program.items[1].kind else {
        panic!("Expected function");
    };
    assert_eq!(func.return_type_id, TypeId::I64);

    assert!(typecheck!("type A = i64; type B = A; fn test() -> B { 42 }").is_ok());

    assert!(typecheck!("type Handle = *opaque; extern fn get() -> Handle;").is_ok());
    assert!(typecheck!("type Bytes = *u8; fn test() -> Bytes { c\"hello\" }").is_ok());

    assert!(typecheck!("type MyInt = i64; struct Point { x: MyInt, y: MyInt }").is_ok());

    assert!(typecheck!("type MyInt = i64; fn add(a: MyInt, b: MyInt) -> MyInt { a + b }").is_ok());

    assert!(
        typecheck!(
            r#"
        struct Point { x: i64, y: i64 }
        type Pos = Point;
        fn test() -> i64 {
            let p: Pos = Pos { x: 1, y: 2 };
            p.x
        }
        "#
        )
        .is_ok()
    );

    assert!(typecheck!("type MyInt = i64; fn test() { let x: MyInt = true; }").is_err());

    assert!(typecheck!("type A = A;").is_err());
    assert!(typecheck!("type A = B; type B = A;").is_err());

    assert!(typecheck!("type A = Undefined;").is_err());
}

#[test]
fn test_fn_type() {
    // fn type as variable
    let result = typecheck!(
        r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 {
            let f: fn(i64, i64) -> i64 = add;
            f(1, 2)
        }
        "#
    );
    assert!(result.is_ok());

    // fn type as parameter
    let result = typecheck!(
        r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn apply(f: fn(i64, i64) -> i64, a: i64, b: i64) -> i64 { f(a, b) }
        fn main() -> i64 { apply(add, 3, 7) }
        "#
    );
    assert!(result.is_ok());

    // no args no return
    let result = typecheck!(
        r#"
        fn noop() {}
        fn call(f: fn()) { f() }
        fn main() { call(noop) }
        "#
    );
    assert!(result.is_ok());

    // signature mismatch
    let result = typecheck!(
        r#"
        fn inc(a: i64) -> i64 { a + 1 }
        fn apply(f: fn(i64, i64) -> i64, a: i64, b: i64) -> i64 { f(a, b) }
        fn main() -> i64 { apply(inc, 1, 2) }
        "#
    );
    assert!(result.is_err());

    // call non func
    let result = typecheck!(
        r#"
        fn main() -> i64 {
            let x: i64 = 42;
            x(1)
        }
        "#
    );
    assert!(result.is_err());

    // fn type in type alias
    let result = typecheck!(
        r#"
        type Callback = fn(i64) -> i64;
        fn double(x: i64) -> i64 { x * 2 }
        fn apply(f: Callback, x: i64) -> i64 { f(x) }
        fn main() -> i64 { apply(double, 21) }
        "#
    );
    assert!(result.is_ok());

    // variadic fn type via type alias
    let result = typecheck!(
        r#"
        extern fn printf(fmt: *u8, ...) -> i64;
        fn main() -> i64 {
            let f: fn(*u8, ...) -> i64 = printf;
            f(c"hello %d", 42)
        }
        "#
    );
    assert!(result.is_ok());
}
