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
        let mut type_checker = TypeChecker::new(&mut type_context);
        type_checker.check_program(&program)
    }};
}

#[test]
fn test_integer_literal() {
    let program = typecheck!("fn main() -> i64 { 42 }").unwrap();

    match &program.items[0].kind {
        ItemKind::Function(func) => {
            let body = func.body.as_ref().expect("Expected function body");
            assert_eq!(body.type_id, TypeId::I64);
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
    let mut type_checker = TypeChecker::new(&mut type_context);
    let program = type_checker.check_program(&program).unwrap();

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
    let mut type_checker = TypeChecker::new(&mut type_context);
    let program = type_checker.check_program(&program).unwrap();

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
            assert_eq!(body.type_id, TypeId::BOOL);
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
fn test_bitwise_operators() {
    let result = typecheck!("fn test() -> i64 { 1 & 2 }");
    assert!(result.is_ok());

    let result = typecheck!("fn test() -> i64 { 1 | 2 }");
    assert!(result.is_ok());
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
