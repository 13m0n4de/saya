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
                    let ty = self.type_ann_to_type(&def.type_ann);
                    self.globals.insert(def.name.clone(), ty);
                }
                Item::Static(def) => {
                    let ty = self.type_ann_to_type(&def.type_ann);
                    self.globals.insert(def.name.clone(), ty);
                }
                Item::Function(def) => {
                    let params = def
                        .params
                        .iter()
                        .map(|param| self.type_ann_to_type(&param.type_ann))
                        .collect();
                    let return_ty = self.type_ann_to_type(&def.return_type_ann);

                    self.functions
                        .insert(def.name.clone(), FunctionSig { params, return_ty });
                }
            }
        }

        let mut typed_items = Vec::new();
        for item in &prog.items {
            match item {
                Item::Const(def) => {
                    let expected_ty = self.type_ann_to_type(&def.type_ann);
                    let typed_init = self.check_expr(&def.init)?;

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
                    let expected_ty = self.type_ann_to_type(&def.type_ann);
                    let typed_init = self.check_expr(&def.init)?;

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
        let return_ty = self.type_ann_to_type(&func.return_type_ann);
        self.current_fn_return_ty = Some(return_ty.clone());

        self.push_scope();

        for param in &func.params {
            let param_ty = self.type_ann_to_type(&param.type_ann);
            self.insert_var(param.name.clone(), param_ty);
        }

        let typed_body = self.check_block(&func.body)?;

        if typed_body.ty != return_ty {
            return Err(TypeError::new(
                format!(
                    "function '{}' has mismatched return type: expected {:?}, found {:?}",
                    func.name, return_ty, typed_body.ty
                ),
                func.body.span,
            ));
        }

        self.pop_scope();
        self.current_fn_return_ty = None;

        Ok(FunctionDef {
            name: func.name.clone(),
            params: func.params.clone(),
            return_type_ann: func.return_type_ann.clone(),
            body: typed_body,
            span: func.span,
        })
    }

    fn type_ann_to_type(&self, ty: &TypeAnn) -> Type {
        match ty {
            TypeAnn::I64 => Type::I64,
            TypeAnn::Str => Type::Str,
            TypeAnn::Array(elem, size) => Type::Array(Box::new(self.type_ann_to_type(elem)), *size),
        }
    }

    fn check_block(&mut self, block: &Block) -> Result<Block<Type>, TypeError> {
        self.push_scope();

        let mut typed_stmts = Vec::new();
        for stmt in &block.stmts {
            typed_stmts.push(self.check_stmt(stmt)?);
        }

        let block_ty = match typed_stmts.last() {
            Some(Stmt {
                kind: StmtKind::Expr(expr),
                ..
            }) => {
                if matches!(expr.kind, ExprKind::Return(_)) {
                    self.current_fn_return_ty.clone().unwrap_or(Type::Unit)
                } else {
                    expr.ty.clone()
                }
            }
            Some(Stmt {
                kind: StmtKind::Semi(expr),
                ..
            }) if matches!(expr.kind, ExprKind::Return(_)) => {
                self.current_fn_return_ty.clone().unwrap_or(Type::Unit)
            }
            _ => Type::Unit,
        };

        self.pop_scope();

        Ok(Block {
            stmts: typed_stmts,
            ty: block_ty,
            span: block.span,
        })
    }

    fn check_stmt(&mut self, stmt: &Stmt<()>) -> Result<Stmt<Type>, TypeError> {
        let kind = match &stmt.kind {
            StmtKind::Let(let_stmt) => {
                let expected_ty = self.type_ann_to_type(&let_stmt.type_ann);
                let typed_init = self.check_expr(&let_stmt.init)?;

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
                let typed_expr = self.check_expr(expr)?;
                StmtKind::Semi(typed_expr)
            }
            StmtKind::Expr(expr) => {
                let typed_expr = self.check_expr(expr)?;
                StmtKind::Expr(typed_expr)
            }
        };

        Ok(Stmt {
            kind,
            span: stmt.span,
        })
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<Expr<Type>, TypeError> {
        let (ty, kind) = match &expr.kind {
            ExprKind::Literal(Literal::Integer(n)) => {
                (Type::I64, ExprKind::Literal(Literal::Integer(*n)))
            }
            ExprKind::Literal(Literal::String(s)) => {
                (Type::Str, ExprKind::Literal(Literal::String(s.clone())))
            }
            ExprKind::Ident(name) => {
                let ty = self
                    .lookup_var(name)
                    .ok_or_else(|| {
                        TypeError::new(format!("undefined variable: {name}"), expr.span)
                    })?
                    .clone();
                (ty, ExprKind::Ident(name.clone()))
            }
            ExprKind::Array(elems) => {
                if elems.is_empty() {
                    return Err(TypeError::new(
                        "cannot infer type of empty array".to_string(),
                        expr.span,
                    ));
                }

                let mut typed_elems = Vec::new();
                let first_elem = self.check_expr(&elems[0])?;
                let elem_ty = first_elem.ty.clone();
                typed_elems.push(first_elem);

                for elem in &elems[1..] {
                    let typed_elem = self.check_expr(elem)?;
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
                (array_ty, ExprKind::Array(typed_elems))
            }
            ExprKind::Repeat(elem, count) => {
                let typed_elem = self.check_expr(elem)?;
                let typed_count = self.check_expr(count)?;

                if typed_count.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!("repeat count must be i64, found {:?}", typed_count.ty),
                        count.span,
                    ));
                }

                if let ExprKind::Literal(Literal::Integer(n)) = typed_count.kind {
                    (
                        Type::Array(Box::new(typed_elem.ty.clone()), n as usize),
                        ExprKind::Repeat(Box::new(typed_elem), Box::new(typed_count)),
                    )
                } else {
                    return Err(TypeError::new(
                        "repeat count must be a constant integer".to_string(),
                        count.span,
                    ));
                }
            }
            ExprKind::Index(array, index) => {
                let typed_array = self.check_expr(array)?;
                let typed_index = self.check_expr(index)?;

                if typed_index.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!("array index must be i64, found {:?}", typed_index.ty),
                        index.span,
                    ));
                }

                match &typed_array.ty {
                    Type::Array(elem_ty, _) => (
                        (**elem_ty).clone(),
                        ExprKind::Index(Box::new(typed_array), Box::new(typed_index)),
                    ),
                    _ => {
                        return Err(TypeError::new(
                            format!("cannot index into type {:?}", typed_array.ty),
                            array.span,
                        ));
                    }
                }
            }
            ExprKind::Call(call) => {
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
                    .ok_or_else(|| {
                        TypeError::new(format!("undefined function: {callee_name}"), call.span)
                    })?
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
                    let typed_arg = self.check_expr(arg)?;
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

                (
                    func_sig.return_ty,
                    ExprKind::Call(Call {
                        callee: Box::new(typed_callee),
                        args: typed_args,
                        span: call.span,
                    }),
                )
            }
            ExprKind::Unary(op, operand) => {
                let typed_operand = self.check_expr(operand)?;

                match op {
                    UnaryOp::Neg => {
                        if typed_operand.ty != Type::I64 {
                            return Err(TypeError::new(
                                format!("cannot negate type {:?}", typed_operand.ty),
                                operand.span,
                            ));
                        }
                        (
                            Type::I64,
                            ExprKind::Unary(op.clone(), Box::new(typed_operand)),
                        )
                    }
                }
            }
            ExprKind::Binary(op, left, right) => {
                let typed_left = self.check_expr(left)?;
                let typed_right = self.check_expr(right)?;

                match op {
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
                        (
                            Type::I64,
                            ExprKind::Binary(
                                op.clone(),
                                Box::new(typed_left),
                                Box::new(typed_right),
                            ),
                        )
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
                        (
                            Type::I64,
                            ExprKind::Binary(
                                op.clone(),
                                Box::new(typed_left),
                                Box::new(typed_right),
                            ),
                        )
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
                        (
                            Type::I64,
                            ExprKind::Binary(
                                op.clone(),
                                Box::new(typed_left),
                                Box::new(typed_right),
                            ),
                        )
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        if typed_left.ty != Type::I64 || typed_right.ty != Type::I64 {
                            return Err(TypeError::new(
                                format!(
                                    "logical operator requires i64 operands, found {:?} and {:?}",
                                    typed_left.ty, typed_right.ty
                                ),
                                expr.span,
                            ));
                        }
                        (
                            Type::I64,
                            ExprKind::Binary(
                                op.clone(),
                                Box::new(typed_left),
                                Box::new(typed_right),
                            ),
                        )
                    }
                }
            }
            ExprKind::Assign(lhs, rhs) => {
                let typed_lhs = self.check_expr(lhs)?;
                let typed_rhs = self.check_expr(rhs)?;

                if typed_lhs.ty != typed_rhs.ty {
                    return Err(TypeError::new(
                        format!(
                            "assignment type mismatch: expected {:?}, found {:?}",
                            typed_lhs.ty, typed_rhs.ty
                        ),
                        expr.span,
                    ));
                }

                (
                    Type::Unit,
                    ExprKind::Assign(Box::new(typed_lhs), Box::new(typed_rhs)),
                )
            }
            ExprKind::Return(val) => {
                let return_ty = self.current_fn_return_ty.clone().ok_or_else(|| {
                    TypeError::new("return outside of function".to_string(), expr.span)
                })?;

                if let Some(v) = val {
                    let typed_val = self.check_expr(v)?;
                    if typed_val.ty != return_ty {
                        return Err(TypeError::new(
                            format!(
                                "return type mismatch: expected {:?}, found {:?}",
                                return_ty, typed_val.ty
                            ),
                            v.span,
                        ));
                    }
                    (Type::Never, ExprKind::Return(Some(Box::new(typed_val))))
                } else {
                    if return_ty != Type::Unit {
                        return Err(TypeError::new(
                            format!("expected return value of type {return_ty:?}"),
                            expr.span,
                        ));
                    }
                    (Type::Never, ExprKind::Return(None))
                }
            }
            ExprKind::Block(block) => {
                let typed_block = self.check_block(block)?;
                let block_ty = typed_block.ty.clone();
                (block_ty, ExprKind::Block(typed_block))
            }
            ExprKind::If(if_expr) => {
                let typed_cond = self.check_expr(&if_expr.cond)?;

                // TODO: Bool
                if typed_cond.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!("if condition must be i64, found {:?}", typed_cond.ty),
                        if_expr.cond.span,
                    ));
                }

                let typed_then = self.check_block(&if_expr.then_body)?;

                let (ty, typed_else) = if let Some(else_expr) = &if_expr.else_body {
                    let typed_else_expr = self.check_expr(else_expr)?;

                    if typed_then.ty != typed_else_expr.ty {
                        return Err(TypeError::new(
                            format!(
                                "if-else branches have different types: {:?} and {:?}",
                                typed_then.ty, typed_else_expr.ty
                            ),
                            else_expr.span,
                        ));
                    }

                    (typed_then.ty.clone(), Some(Box::new(typed_else_expr)))
                } else {
                    (Type::Unit, None)
                };

                (
                    ty,
                    ExprKind::If(If {
                        cond: Box::new(typed_cond),
                        then_body: Box::new(typed_then),
                        else_body: typed_else,
                        span: if_expr.span,
                    }),
                )
            }
            ExprKind::While(while_expr) => {
                let typed_cond = self.check_expr(&while_expr.cond)?;

                // TODO: Bool
                if typed_cond.ty != Type::I64 {
                    return Err(TypeError::new(
                        format!("while condition must be i64, found {:?}", typed_cond.ty),
                        while_expr.cond.span,
                    ));
                }

                let typed_body = self.check_block(&while_expr.body)?;

                (
                    Type::Unit,
                    ExprKind::While(While {
                        cond: Box::new(typed_cond),
                        body: Box::new(typed_body),
                        span: while_expr.span,
                    }),
                )
            }
            ExprKind::Break => (Type::Never, ExprKind::Break),
            ExprKind::Continue => (Type::Never, ExprKind::Continue),
        };

        Ok(Expr {
            kind,
            ty,
            span: expr.span,
        })
    }
}
