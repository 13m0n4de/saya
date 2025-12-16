use saya::ast::*;
use saya::lexer::Lexer;
use saya::parser::Parser;

macro_rules! parse {
    ($input:expr) => {{
        let lexer = Lexer::new($input);
        let mut parser = Parser::new(lexer).unwrap();
        parser.parse()
    }};
}

macro_rules! parse_expr {
    ($input:expr) => {{
        let lexer = Lexer::new($input);
        let mut parser = Parser::new(lexer).unwrap();
        parser.parse_expression()
    }};
}

#[test]
fn test_integer_literal() {
    let expr = parse_expr!("42").unwrap();
    assert!(matches!(expr.kind, ExprKind::Literal(Literal::Integer(42))));
}

#[test]
fn test_string_literal() {
    let expr = parse_expr!(r#""hello""#).unwrap();
    assert!(matches!(
        expr.kind,
        ExprKind::Literal(Literal::String(s)) if s == "hello"
    ));
}

#[test]
fn test_bool_literal() {
    let expr_true = parse_expr!("true").unwrap();
    let expr_false = parse_expr!("false").unwrap();

    assert!(matches!(
        expr_true.kind,
        ExprKind::Literal(Literal::Bool(true))
    ));
    assert!(matches!(
        expr_false.kind,
        ExprKind::Literal(Literal::Bool(false))
    ));
}

#[test]
fn test_struct_literal() {
    let expr =
        parse_expr!("Line { start: Point { x: 0, y: 0 }, end: Point { x: 10, y: 10 } }").unwrap();

    match expr.kind {
        ExprKind::Struct(struct_expr) => {
            assert_eq!(struct_expr.name, "Line");
            assert_eq!(struct_expr.fields.len(), 2);

            assert_eq!(struct_expr.fields[0].name, "start");
            match &struct_expr.fields[0].value.kind {
                ExprKind::Struct(start_point) => {
                    assert_eq!(start_point.name, "Point");
                    assert_eq!(start_point.fields.len(), 2);

                    // start.x = 0
                    assert_eq!(start_point.fields[0].name, "x");
                    assert!(matches!(
                        start_point.fields[0].value.kind,
                        ExprKind::Literal(Literal::Integer(0))
                    ));

                    // start.y = 0
                    assert_eq!(start_point.fields[1].name, "y");
                    assert!(matches!(
                        start_point.fields[1].value.kind,
                        ExprKind::Literal(Literal::Integer(0))
                    ));
                }
                _ => panic!("Expected nested struct literal for 'start'"),
            }

            assert_eq!(struct_expr.fields[1].name, "end");
            match &struct_expr.fields[1].value.kind {
                ExprKind::Struct(end_point) => {
                    assert_eq!(end_point.name, "Point");
                    assert_eq!(end_point.fields.len(), 2);

                    // end.x = 10
                    assert_eq!(end_point.fields[0].name, "x");
                    assert!(matches!(
                        end_point.fields[0].value.kind,
                        ExprKind::Literal(Literal::Integer(10))
                    ));

                    // end.y = 10
                    assert_eq!(end_point.fields[1].name, "y");
                    assert!(matches!(
                        end_point.fields[1].value.kind,
                        ExprKind::Literal(Literal::Integer(10))
                    ));
                }
                _ => panic!("Expected nested struct literal for 'end'"),
            }
        }
        _ => panic!("Expected struct literal"),
    }
}

#[test]
fn test_operator_precedence() {
    let expr = parse_expr!("1 + 2 * 3").unwrap();

    match expr.kind {
        ExprKind::Binary(BinaryOp::Add, left, right) => {
            assert!(matches!(left.kind, ExprKind::Literal(Literal::Integer(1))));
            assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
        }
        _ => panic!("Expected Add at top level"),
    }
}

#[test]
fn test_comparison_precedence() {
    let expr = parse_expr!("1 + 2 < 3 * 4").unwrap();

    match expr.kind {
        ExprKind::Binary(BinaryOp::Lt, left, right) => {
            assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::Add, _, _)));
            assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
        }
        _ => panic!("Expected Lt at top level"),
    }
}

#[test]
fn test_logical_operators() {
    let expr = parse_expr!("a && b || c").unwrap();

    match expr.kind {
        ExprKind::Binary(BinaryOp::Or, left, _right) => {
            assert!(matches!(left.kind, ExprKind::Binary(BinaryOp::And, _, _)));
        }
        _ => panic!("Expected Or at top level"),
    }
}

#[test]
fn test_unary_operators() {
    let neg = parse_expr!("-42").unwrap();
    let not = parse_expr!("!true").unwrap();

    assert!(matches!(neg.kind, ExprKind::Unary(UnaryOp::Neg, _)));
    assert!(matches!(not.kind, ExprKind::Unary(UnaryOp::Not, _)));
}

#[test]
fn test_array_literal() {
    let expr = parse_expr!("[1, 2, 3]").unwrap();

    match expr.kind {
        ExprKind::Array(elements) => {
            assert_eq!(elements.len(), 3);
        }
        _ => panic!("Expected array literal"),
    }
}

#[test]
fn test_array_repeat() {
    let expr = parse_expr!("[42; 5]").unwrap();

    assert!(matches!(expr.kind, ExprKind::Repeat(_, _)));
}

#[test]
fn test_array_index() {
    let expr = parse_expr!("arr[0]").unwrap();

    assert!(matches!(expr.kind, ExprKind::Index(_, _)));
}

#[test]
fn test_function_call() {
    let expr = parse_expr!("foo(1, 2)").unwrap();

    match expr.kind {
        ExprKind::Call(call) => {
            assert!(matches!(&call.callee.kind, ExprKind::Ident(name) if name == "foo"));
            assert_eq!(call.args.len(), 2);
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_function_call_no_args() {
    let expr = parse_expr!("foo()").unwrap();

    match expr.kind {
        ExprKind::Call(call) => {
            assert!(matches!(&call.callee.kind, ExprKind::Ident(name) if name == "foo"));
            assert_eq!(call.args.len(), 0);
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_let_binding() {
    let program = parse!("fn test() -> i64 { let x: i64 = 42; x }").unwrap();

    match &program.items[0] {
        Item::Function(func) => match &func.body.stmts[0].kind {
            StmtKind::Let(let_stmt) => {
                assert_eq!(let_stmt.name, "x");
                assert_eq!(let_stmt.type_ann.kind, TypeAnnKind::I64);
            }
            _ => panic!("Expected let statement"),
        },
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_assignment() {
    let expr = parse_expr!("x = 42").unwrap();

    assert!(matches!(expr.kind, ExprKind::Assign(_, _)));
}

#[test]
fn test_if_else() {
    let expr = parse_expr!("if true { 1 } else { 2 }").unwrap();

    match expr.kind {
        ExprKind::If(if_expr) => {
            assert!(if_expr.else_body.is_some());
        }
        _ => panic!("Expected if expression"),
    }
}

#[test]
fn test_if_without_else() {
    let expr = parse_expr!("if true { 1 }").unwrap();

    match expr.kind {
        ExprKind::If(if_expr) => {
            assert!(if_expr.else_body.is_none());
        }
        _ => panic!("Expected if expression"),
    }
}

#[test]
fn test_while_loop() {
    let expr = parse_expr!("while true { break; }").unwrap();

    assert!(matches!(expr.kind, ExprKind::While(_)));
}

#[test]
fn test_break_continue() {
    let break_expr = parse_expr!("break").unwrap();
    let continue_expr = parse_expr!("continue").unwrap();

    assert!(matches!(break_expr.kind, ExprKind::Break));
    assert!(matches!(continue_expr.kind, ExprKind::Continue));
}

#[test]
fn test_return() {
    let return_with_value = parse_expr!("return 42").unwrap();
    let return_without_value = parse_expr!("return;").unwrap();

    assert!(matches!(return_with_value.kind, ExprKind::Return(Some(_))));
    assert!(matches!(return_without_value.kind, ExprKind::Return(None)));
}

#[test]
fn test_block_with_trailing_expr() {
    let expr = parse_expr!("{ let x: i64 = 1; x + 1 }").unwrap();

    match expr.kind {
        ExprKind::Block(block) => {
            assert_eq!(block.stmts.len(), 2);
            assert!(matches!(block.stmts[0].kind, StmtKind::Let(_)));
            assert!(matches!(block.stmts[1].kind, StmtKind::Expr(_)));
        }
        _ => panic!("Expected block"),
    }
}

#[test]
fn test_block_all_semicolons() {
    let expr = parse_expr!("{ 1; 2; }").unwrap();

    match expr.kind {
        ExprKind::Block(block) => {
            assert!(matches!(block.stmts[0].kind, StmtKind::Semi(_)));
            assert!(matches!(block.stmts[1].kind, StmtKind::Semi(_)));
        }
        _ => panic!("Expected block"),
    }
}

#[test]
fn test_function_definition() {
    let program = parse!("fn add(a: i64, b: i64) -> i64 { a + b }").unwrap();

    match &program.items[0] {
        Item::Function(func) => {
            assert_eq!(func.name, "add");
            assert_eq!(func.params.len(), 2);
            assert_eq!(func.params[0].name, "a");
            assert_eq!(func.params[1].name, "b");
            assert_eq!(func.return_type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_const_definition() {
    let program = parse!("const PI: i64 = 3;").unwrap();

    match &program.items[0] {
        Item::Const(const_def) => {
            assert_eq!(const_def.name, "PI");
            assert_eq!(const_def.type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected const"),
    }
}

#[test]
fn test_static_definition() {
    let program = parse!("static GLOBAL: i64 = 42;").unwrap();

    match &program.items[0] {
        Item::Static(static_def) => {
            assert_eq!(static_def.name, "GLOBAL");
            assert_eq!(static_def.type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected static"),
    }
}

#[test]
fn test_struct_definition() {
    let program = parse!("struct Position { x: i64, y: i64 }").unwrap();

    match &program.items[0] {
        Item::Struct(struct_def) => {
            assert_eq!(struct_def.name, "Position");
            assert_eq!(struct_def.fields[0].name, "x");
            assert_eq!(struct_def.fields[0].type_ann.kind, TypeAnnKind::I64);
            assert_eq!(struct_def.fields[1].name, "y");
            assert_eq!(struct_def.fields[1].type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected struct"),
    }
}

#[test]
fn test_multiple_items() {
    let program = parse!(
        r#"
        const X: i64 = 1;
        fn foo() -> i64 { 42 }
        static Y: i64 = 2;
    "#
    )
    .unwrap();

    assert_eq!(program.items.len(), 3);
    assert!(matches!(program.items[0], Item::Const(_)));
    assert!(matches!(program.items[1], Item::Function(_)));
    assert!(matches!(program.items[2], Item::Static(_)));
}

#[test]
fn test_array_type() {
    let program = parse!("fn test() -> i64 { let arr: [i64; 3] = [1, 2, 3]; 0 }").unwrap();

    match &program.items[0] {
        Item::Function(func) => match &func.body.stmts[0].kind {
            StmtKind::Let(let_stmt) => {
                assert!(matches!(let_stmt.type_ann.kind, TypeAnnKind::Array(_, 3)));
            }
            _ => panic!("Expected let statement"),
        },
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_missing_semicolon() {
    let result = parse!("fn test() -> i64 { let x: i64 = 42 x }");
    assert!(result.is_err(), "should fail on missing semicolon");
}

#[test]
fn test_unmatched_paren() {
    let result = parse!("fn test() -> i64 { (1 + 2 }");
    assert!(result.is_err(), "should fail on unmatched paren");
}

#[test]
fn test_invalid_syntax() {
    let result = parse!("fn test() -> i64 { let = 42; }");
    assert!(result.is_err(), "should fail on invalid syntax");
}

#[test]
fn test_chained_calls_and_index() {
    let expr = parse_expr!("foo(1)[0]").unwrap();

    match expr.kind {
        ExprKind::Index(array, _idx) => {
            assert!(matches!(array.kind, ExprKind::Call(_)));
        }
        _ => panic!("Expected index of call"),
    }
}

#[test]
fn test_extern_declarations() {
    let program = parse!(
        r#"
        extern static stderr: i64;
        extern fn puts(s: [u8]) -> i64;
    "#
    )
    .unwrap();

    assert_eq!(program.items.len(), 2);

    match &program.items[0] {
        Item::Extern(ExternItem::Static(static_decl)) => {
            assert_eq!(static_decl.name, "stderr");
            assert_eq!(static_decl.type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected extern static"),
    }

    match &program.items[1] {
        Item::Extern(ExternItem::Function(func)) => {
            assert_eq!(func.name, "puts");
            assert_eq!(func.params.len(), 1);
            assert_eq!(func.params[0].name, "s");
            assert_eq!(func.return_type_ann.kind, TypeAnnKind::I64);
        }
        _ => panic!("Expected extern function"),
    }
}
