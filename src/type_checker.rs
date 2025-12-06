use std::{collections::HashMap, error::Error, fmt};

use crate::{ast::*, span::Span};

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

impl TypeError {
    pub fn new(message: String, span: Span) -> Self {
        Self { message, span }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "type error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl Error for TypeError {}

#[derive(Default)]
pub struct TypeChecker {
    scopes: Vec<HashMap<String, Type>>,
    functions: HashMap<String, FunctionSig>,
    globals: HashMap<String, Type>,
    current_fn_return_ty: Option<Type>,
}

#[derive(Clone)]
struct FunctionSig {
    params: Vec<Type>,
    return_ty: Type,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            functions: HashMap::new(),
            globals: HashMap::new(),
            current_fn_return_ty: None,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup_var(&self, name: &str) -> Option<&Type> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .or_else(|| self.globals.get(name))
    }

    fn insert_var(&mut self, name: String, ty: Type) {
        self.scopes
            .last_mut()
            .expect("ICE: scope stack should not be empty")
            .insert(name, ty);
    }

    pub fn check_program(&mut self, prog: &Program) -> Result<Program<Type>, TypeError> {
        for item in &prog.items {
            match item {
                Item::Const(def) => {
                    let ty = Type::from(&def.type_ann);
                    self.globals.insert(def.name.clone(), ty);
                }
                Item::Static(def) => {
                    let ty = Type::from(&def.type_ann);
                    self.globals.insert(def.name.clone(), ty);
                }
                Item::Function(def) => {
                    let params = def
                        .params
                        .iter()
                        .map(|param| Type::from(&param.type_ann))
                        .collect();
                    let return_ty = Type::from(&def.return_type_ann);

                    self.functions
                        .insert(def.name.clone(), FunctionSig { params, return_ty });
                }
            }
        }

        let mut typed_items = Vec::new();
        for item in &prog.items {
            match item {
                Item::Const(def) => {
                    let expected_ty = Type::from(&def.type_ann);
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.ty != expected_ty {
                        return Err(TypeError::new(
                            format!(
                                "const '{}' type mismatch: expected {:?}, found {:?}",
                                def.name, expected_ty, typed_init.ty
                            ),
                            def.init.span,
                        ));
                    }

                    typed_items.push(Item::Const(ConstDef {
                        name: def.name.clone(),
                        type_ann: def.type_ann.clone(),
                        init: Box::new(typed_init),
                        span: def.span,
                    }));
                }
                Item::Static(def) => {
                    let expected_ty = Type::from(&def.type_ann);
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.ty != expected_ty {
                        return Err(TypeError::new(
                            format!(
                                "static '{}' type mismatch: expected {:?}, found {:?}",
                                def.name, expected_ty, typed_init.ty
                            ),
                            def.init.span,
                        ));
                    }

                    typed_items.push(Item::Static(StaticDef {
                        name: def.name.clone(),
                        type_ann: def.type_ann.clone(),
                        init: Box::new(typed_init),
                        span: def.span,
                    }));
                }
                Item::Function(def) => {
                    let typed_func = self.check_function(def)?;
                    typed_items.push(Item::Function(typed_func));
                }
            }
        }

        Ok(Program { items: typed_items })
    }

    fn check_function(&mut self, func: &FunctionDef<()>) -> Result<FunctionDef<Type>, TypeError> {
        let return_ty = Type::from(&func.return_type_ann);

        // If function has no body, it's a external function
        let Some(body) = &func.body else {
            return Ok(FunctionDef {
                name: func.name.clone(),
                params: func.params.clone(),
                return_type_ann: func.return_type_ann.clone(),
                body: None,
                span: func.span,
            });
        };

        self.current_fn_return_ty = Some(return_ty.clone());
        self.push_scope();

        for param in &func.params {
            let param_ty = Type::from(&param.type_ann);
            self.insert_var(param.name.clone(), param_ty);
        }

        let typed_body = self.check_block(body)?;

        // The `Never` type is compatible with any return type
        if typed_body.ty != return_ty && typed_body.ty != Type::Never {
            return Err(TypeError::new(
                format!(
                    "function '{}' has mismatched return type: expected {:?}, found {:?}",
                    func.name, return_ty, typed_body.ty
                ),
                body.span,
            ));
        }

        self.pop_scope();
        self.current_fn_return_ty = None;

        Ok(FunctionDef {
            name: func.name.clone(),
            params: func.params.clone(),
            return_type_ann: func.return_type_ann.clone(),
            body: Some(typed_body),
            span: func.span,
        })
    }

    fn check_block(&mut self, block: &Block) -> Result<Block<Type>, TypeError> {
        self.push_scope();

        let mut typed_stmts = Vec::new();
        let mut has_never = false;

        for (idx, stmt) in block.stmts.iter().enumerate() {
            if has_never {
                return Err(TypeError::new(
                    "unreachable statement after diverging expression".to_string(),
                    stmt.span,
                ));
            }

            let typed_stmt = self.check_statement(stmt)?;
            let is_last = idx == block.stmts.len() - 1;

            match &typed_stmt.kind {
                StmtKind::Expr(expr)
                    if !is_last && expr.ty != Type::Unit && expr.ty != Type::Never =>
                {
                    return Err(TypeError::new(
                        format!(
                            "expected `;` after expression: expected type `()`, found type `{:?}`",
                            expr.ty
                        ),
                        expr.span,
                    ));
                }
                StmtKind::Expr(expr) | StmtKind::Semi(expr) if expr.ty == Type::Never => {
                    has_never = true;
                }
                _ => {}
            }

            typed_stmts.push(typed_stmt);
        }

        let block_ty = match typed_stmts.last() {
            Some(Stmt {
                kind: StmtKind::Expr(expr),
                ..
            }) => expr.ty.clone(),
            Some(Stmt {
                kind: StmtKind::Semi(expr),
                ..
            }) if expr.ty == Type::Never => Type::Never,
            _ => Type::Unit,
        };

        self.pop_scope();

        Ok(Block {
            stmts: typed_stmts,
            ty: block_ty,
            span: block.span,
        })
    }

    fn check_statement(&mut self, stmt: &Stmt<()>) -> Result<Stmt<Type>, TypeError> {
        let kind = match &stmt.kind {
            StmtKind::Let(let_stmt) => {
                let expected_ty = Type::from(&let_stmt.type_ann);
                let typed_init = self.check_expression(&let_stmt.init)?;

                if typed_init.ty != expected_ty {
                    return Err(TypeError::new(
                        format!(
                            "type mismatch in let binding: expected {:?}, found {:?}",
                            expected_ty, typed_init.ty
                        ),
                        let_stmt.init.span,
                    ));
                }

                self.insert_var(let_stmt.name.clone(), expected_ty);

                StmtKind::Let(Let {
                    name: let_stmt.name.clone(),
                    type_ann: let_stmt.type_ann.clone(),
                    init: typed_init,
                    span: let_stmt.span,
                })
            }
            StmtKind::Semi(expr) => {
                let typed_expr = self.check_expression(expr)?;
                StmtKind::Semi(typed_expr)
            }
            StmtKind::Expr(expr) => {
                let typed_expr = self.check_expression(expr)?;
                StmtKind::Expr(typed_expr)
            }
        };

        Ok(Stmt {
            kind,
            span: stmt.span,
        })
    }

    fn check_expression(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        match &expr.kind {
            ExprKind::Literal(..) => Ok(Self::check_expr_literal(expr)),
            ExprKind::Ident(..) => self.check_expr_ident(expr),
            ExprKind::Array(..) => self.check_expr_array(expr),
            ExprKind::Repeat(..) => self.check_expr_repeat(expr),
            ExprKind::Index(..) => self.check_expr_index(expr),
            ExprKind::Call(..) => self.check_expr_call(expr),
            ExprKind::Unary(..) => self.check_expr_unary(expr),
            ExprKind::Binary(..) => self.check_expr_binary(expr),
            ExprKind::Assign(..) => self.check_expr_assign(expr),
            ExprKind::Return(..) => self.check_expr_return(expr),
            ExprKind::Block(..) => self.check_expr_block(expr),
            ExprKind::If(..) => self.check_expr_if(expr),
            ExprKind::While(..) => self.check_expr_while(expr),
            ExprKind::Break => Ok(Expr {
                kind: ExprKind::Break,
                ty: Type::Never,
                span: expr.span,
            }),
            ExprKind::Continue => Ok(Expr {
                kind: ExprKind::Continue,
                ty: Type::Never,
                span: expr.span,
            }),
        }
    }

    fn check_expr_literal(expr: &Expr) -> Expr<Type> {
        let ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        let (ty, kind) = match lit {
            Literal::Integer(n) => (Type::I64, ExprKind::Literal(Literal::Integer(*n))),
            Literal::String(s) => (Type::Str, ExprKind::Literal(Literal::String(s.clone()))),
            Literal::Bool(b) => (Type::Bool, ExprKind::Literal(Literal::Bool(*b))),
        };

        Expr {
            kind,
            ty,
            span: expr.span,
        }
    }

    fn check_expr_ident(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Ident(name) = &expr.kind else {
            unreachable!()
        };

        let ty = self
            .lookup_var(name)
            .ok_or_else(|| TypeError::new(format!("undefined variable: {name}"), expr.span))?
            .clone();

        Ok(Expr {
            kind: ExprKind::Ident(name.clone()),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_array(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Array(elems) = &expr.kind else {
            unreachable!()
        };

        if elems.is_empty() {
            return Err(TypeError::new(
                "cannot infer type of empty array".to_string(),
                expr.span,
            ));
        }

        let mut typed_elems = Vec::new();
        let first_elem = self.check_expression(&elems[0])?;
        let elem_ty = first_elem.ty.clone();
        typed_elems.push(first_elem);

        for elem in &elems[1..] {
            let typed_elem = self.check_expression(elem)?;
            if typed_elem.ty != elem_ty {
                return Err(TypeError::new(
                    format!(
                        "array element type mismatch: expected {:?}, found {:?}",
                        elem_ty, typed_elem.ty
                    ),
                    elem.span,
                ));
            }
            typed_elems.push(typed_elem);
        }

        let array_ty = Type::Array(Box::new(elem_ty), elems.len());
        Ok(Expr {
            kind: ExprKind::Array(typed_elems),
            ty: array_ty,
            span: expr.span,
        })
    }

    fn check_expr_repeat(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let typed_elem = self.check_expression(elem)?;
        let typed_count = self.check_expression(count)?;

        if typed_count.ty != Type::I64 {
            return Err(TypeError::new(
                format!("repeat count must be i64, found {:?}", typed_count.ty),
                count.span,
            ));
        }

        if let ExprKind::Literal(Literal::Integer(n)) = typed_count.kind {
            Ok(Expr {
                kind: ExprKind::Repeat(Box::new(typed_elem.clone()), Box::new(typed_count)),
                ty: Type::Array(Box::new(typed_elem.ty), n as usize),
                span: expr.span,
            })
        } else {
            Err(TypeError::new(
                "repeat count must be a constant integer".to_string(),
                count.span,
            ))
        }
    }

    fn check_expr_index(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Index(array, index) = &expr.kind else {
            unreachable!()
        };

        let typed_array = self.check_expression(array)?;
        let typed_index = self.check_expression(index)?;

        if typed_index.ty != Type::I64 {
            return Err(TypeError::new(
                format!("array index must be i64, found {:?}", typed_index.ty),
                index.span,
            ));
        }

        match typed_array.ty {
            Type::Array(ref elem_ty, _) => Ok(Expr {
                kind: ExprKind::Index(Box::new(typed_array.clone()), Box::new(typed_index)),
                ty: *elem_ty.clone(),
                span: expr.span,
            }),
            _ => Err(TypeError::new(
                format!("cannot index into type {:?}", typed_array.ty),
                array.span,
            )),
        }
    }

    fn check_expr_call(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Call(call) = &expr.kind else {
            unreachable!()
        };

        let callee_name = if let ExprKind::Ident(name) = &call.callee.kind {
            name.clone()
        } else {
            return Err(TypeError::new(
                "function call must use identifier".to_string(),
                call.callee.span,
            ));
        };

        let func_sig = self
            .functions
            .get(&callee_name)
            .ok_or_else(|| TypeError::new(format!("undefined function: {callee_name}"), call.span))?
            .clone();

        if call.args.len() != func_sig.params.len() {
            return Err(TypeError::new(
                format!(
                    "function '{callee_name}' expects {} arguments, got {}",
                    func_sig.params.len(),
                    call.args.len()
                ),
                call.span,
            ));
        }

        let mut typed_args = Vec::new();
        for (arg, param_ty) in call.args.iter().zip(func_sig.params) {
            let typed_arg = self.check_expression(arg)?;
            if typed_arg.ty != param_ty {
                return Err(TypeError::new(
                    format!(
                        "argument type mismatch: expected {:?}, found {:?}",
                        param_ty, typed_arg.ty
                    ),
                    arg.span,
                ));
            }
            typed_args.push(typed_arg);
        }

        let typed_callee = Expr {
            kind: ExprKind::Ident(callee_name),
            ty: func_sig.return_ty.clone(),
            span: call.callee.span,
        };

        Ok(Expr {
            kind: ExprKind::Call(Call {
                callee: Box::new(typed_callee),
                args: typed_args,
                span: call.span,
            }),
            ty: func_sig.return_ty,
            span: expr.span,
        })
    }

    fn check_expr_unary(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Unary(op, operand) = &expr.kind else {
            unreachable!()
        };

        let typed_operand = self.check_expression(operand)?;

        let ty = match op {
            UnaryOp::Neg => {
                if typed_operand.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!("cannot apply `-` to type {:?}", typed_operand.ty),
                        operand.span,
                    ));
                }
                Type::I64
            }
            UnaryOp::Not => match typed_operand.ty {
                Type::Bool => Type::Bool,
                Type::I64 => Type::I64,
                _ => {
                    return Err(TypeError::new(
                        format!("cannot apply `!` to type {:?}", typed_operand.ty),
                        operand.span,
                    ));
                }
            },
        };

        Ok(Expr {
            kind: ExprKind::Unary(op.clone(), Box::new(typed_operand)),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_binary(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Binary(op, left, right) = &expr.kind else {
            unreachable!()
        };

        let typed_left = self.check_expression(left)?;
        let typed_right = self.check_expression(right)?;

        let ty = match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Rem
            | BinaryOp::BitAnd
            | BinaryOp::BitOr => {
                if typed_left.ty != Type::I64 || typed_right.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!(
                            "arithmetic operator requires i64 operands, found {:?} and {:?}",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::I64
            }
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if typed_left.ty != Type::I64 || typed_right.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!(
                            "comparison operator requires i64 operands, found {:?} and {:?}",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::Bool
            }
            BinaryOp::Eq | BinaryOp::Ne => {
                if typed_left.ty != typed_right.ty {
                    return Err(TypeError::new(
                        format!(
                            "equality operator requires same types, found {:?} and {:?}",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::Bool
            }
            BinaryOp::And | BinaryOp::Or => {
                if typed_left.ty != Type::Bool || typed_right.ty != Type::Bool {
                    return Err(TypeError::new(
                        format!(
                            "logical operator requires bool operands, found {:?} and {:?}",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::Bool
            }
        };

        Ok(Expr {
            kind: ExprKind::Binary(op.clone(), Box::new(typed_left), Box::new(typed_right)),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_assign(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let typed_lhs = self.check_expression(lhs)?;
        let typed_rhs = self.check_expression(rhs)?;

        if typed_lhs.ty != typed_rhs.ty {
            return Err(TypeError::new(
                format!(
                    "assignment type mismatch: expected {:?}, found {:?}",
                    typed_lhs.ty, typed_rhs.ty
                ),
                expr.span,
            ));
        }

        Ok(Expr {
            kind: ExprKind::Assign(Box::new(typed_lhs), Box::new(typed_rhs)),
            ty: Type::Unit,
            span: expr.span,
        })
    }

    fn check_expr_return(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Return(val) = &expr.kind else {
            unreachable!()
        };

        let return_ty = self
            .current_fn_return_ty
            .clone()
            .ok_or_else(|| TypeError::new("return outside of function".to_string(), expr.span))?;

        let kind = if let Some(v) = val {
            let typed_val = self.check_expression(v)?;
            if typed_val.ty != return_ty {
                return Err(TypeError::new(
                    format!(
                        "return type mismatch: expected {:?}, found {:?}",
                        return_ty, typed_val.ty
                    ),
                    v.span,
                ));
            }
            ExprKind::Return(Some(Box::new(typed_val)))
        } else {
            if return_ty != Type::Unit {
                return Err(TypeError::new(
                    format!("expected return value of type {return_ty:?}"),
                    expr.span,
                ));
            }
            ExprKind::Return(None)
        };

        Ok(Expr {
            kind,
            ty: Type::Never,
            span: expr.span,
        })
    }

    fn check_expr_block(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::Block(block) = &expr.kind else {
            unreachable!()
        };

        let typed_block = self.check_block(block)?;
        let block_ty = typed_block.ty.clone();

        Ok(Expr {
            kind: ExprKind::Block(typed_block),
            ty: block_ty,
            span: expr.span,
        })
    }

    fn check_expr_if(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&if_expr.cond)?;

        if typed_cond.ty != Type::Bool {
            return Err(TypeError::new(
                format!("if condition must be bool, found {:?}", typed_cond.ty),
                if_expr.cond.span,
            ));
        }

        let typed_then = self.check_block(&if_expr.then_body)?;

        let (ty, typed_else) = if let Some(else_expr) = &if_expr.else_body {
            let typed_else_expr = self.check_expression(else_expr)?;

            let result_ty = match (&typed_then.ty, &typed_else_expr.ty) {
                (Type::Never, Type::Never) => Type::Never,
                (Type::Never, other) | (other, Type::Never) => other.clone(),
                (then_ty, else_ty) if then_ty == else_ty => then_ty.clone(),
                (then_ty, else_ty) => {
                    return Err(TypeError::new(
                        format!(
                            "if-else branches have different types: {then_ty:?} and {else_ty:?}",
                        ),
                        else_expr.span,
                    ));
                }
            };

            (result_ty, Some(Box::new(typed_else_expr)))
        } else {
            // when there is no else-branch, the implicit else branch returns `Unit`,
            // so the entire `Expr` is always `Unit`
            // (even if the then-branch is `Never`)
            (Type::Unit, None)
        };

        Ok(Expr {
            kind: ExprKind::If(If {
                cond: Box::new(typed_cond),
                then_body: Box::new(typed_then),
                else_body: typed_else,
                span: if_expr.span,
            }),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_while(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&while_expr.cond)?;

        if typed_cond.ty != Type::Bool {
            return Err(TypeError::new(
                format!("while condition must be bool, found {:?}", typed_cond.ty),
                while_expr.cond.span,
            ));
        }

        let typed_body = self.check_block(&while_expr.body)?;

        Ok(Expr {
            kind: ExprKind::While(While {
                cond: Box::new(typed_cond),
                body: Box::new(typed_body),
                span: while_expr.span,
            }),
            ty: Type::Unit,
            span: expr.span,
        })
    }
}
