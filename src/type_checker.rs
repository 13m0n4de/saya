use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use crate::{
    ast, hir,
    span::Span,
    ty::{FieldInfo, Type, TypeKind},
};

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
    constants: HashMap<String, Type>,
    current_fn_return_ty: Option<Type>,
    types: HashMap<String, Type>,
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
            constants: HashMap::new(),
            current_fn_return_ty: None,
            types: HashMap::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup_var_type(&self, name: &str) -> Option<&Type> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .or_else(|| self.globals.get(name))
            .or_else(|| self.constants.get(name))
    }

    fn insert_local_var(&mut self, name: String, ty: Type) {
        self.scopes
            .last_mut()
            .expect("ICE: scope stack should not be empty")
            .insert(name, ty);
    }

    fn resolve_type(&mut self, type_ann: &ast::TypeAnn) -> Result<Type, TypeError> {
        match &type_ann.kind {
            ast::TypeAnnKind::I64 => Ok(Type::i64()),
            ast::TypeAnnKind::U8 => Ok(Type::u8()),
            ast::TypeAnnKind::Bool => Ok(Type::bool()),
            ast::TypeAnnKind::Unit => Ok(Type::unit()),
            ast::TypeAnnKind::Never => Ok(Type::never()),

            ast::TypeAnnKind::Pointer(inner) => {
                let inner_ty = self.resolve_type(inner)?;
                Ok(Type::pointer(inner_ty))
            }

            ast::TypeAnnKind::Array(elem, count) => {
                let elem_ty = self.resolve_type(elem)?;
                Ok(Type::array(elem_ty, *count))
            }

            ast::TypeAnnKind::Slice(elem) => {
                let elem_ty = self.resolve_type(elem)?;
                Ok(Type::slice(elem_ty))
            }

            ast::TypeAnnKind::Named(name) => {
                self.types.get(name).cloned().ok_or_else(|| {
                    TypeError::new(format!("undefined type `{name}`"), type_ann.span)
                })
            }
        }
    }

    pub fn check_program(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        self.collect_types(prog)?;
        self.resolve_types(prog)?;
        self.collect_signatures(prog)?;
        self.check_items(prog)
    }

    fn collect_types(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            if let ast::Item::Struct(def) = item {
                if self.types.contains_key(&def.name) {
                    return Err(TypeError::new(
                        format!("duplicate struct definition `{}`", def.name),
                        def.span,
                    ));
                }
                self.types.insert(def.name.clone(), Type::unit());
            }
        }
        Ok(())
    }

    fn resolve_types(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            if let ast::Item::Struct(def) = item {
                self.resolve_struct(def)?;
            }
        }
        Ok(())
    }

    fn collect_signatures(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            match item {
                ast::Item::Const(def) => {
                    let ty = self.resolve_type(&def.type_ann)?;
                    self.constants.insert(def.name.clone(), ty);
                }
                ast::Item::Static(def) => {
                    let ty = self.resolve_type(&def.type_ann)?;
                    self.globals.insert(def.name.clone(), ty);
                }
                ast::Item::Function(def) => {
                    let params = def
                        .params
                        .iter()
                        .map(|param| self.resolve_type(&param.type_ann))
                        .collect::<Result<Vec<_>, _>>()?;
                    let return_ty = self.resolve_type(&def.return_type_ann)?;

                    self.functions
                        .insert(def.name.clone(), FunctionSig { params, return_ty });
                }
                ast::Item::Extern(extern_item) => match extern_item {
                    ast::ExternItem::Static(decl) => {
                        let ty = self.resolve_type(&decl.type_ann)?;
                        self.globals.insert(decl.name.clone(), ty);
                    }
                    ast::ExternItem::Function(decl) => {
                        let params = decl
                            .params
                            .iter()
                            .map(|param| self.resolve_type(&param.type_ann))
                            .collect::<Result<Vec<_>, _>>()?;
                        let return_ty = self.resolve_type(&decl.return_type_ann)?;

                        self.functions
                            .insert(decl.name.clone(), FunctionSig { params, return_ty });
                    }
                },
                ast::Item::Struct(_) => {}
            }
        }
        Ok(())
    }

    fn check_items(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        let mut typed_items = Vec::new();
        for item in &prog.items {
            let typed_item = match item {
                ast::Item::Const(def) => {
                    let declared_ty = self.resolve_type(&def.type_ann)?;
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.ty != declared_ty {
                        return Err(TypeError::new(
                            format!(
                                "type mismatch in const `{}`: expected `{:?}`, found `{:?}`",
                                def.name, declared_ty, typed_init.ty
                            ),
                            def.init.span,
                        ));
                    }

                    hir::Item::Const(hir::ConstDef {
                        name: def.name.clone(),
                        ty: declared_ty,
                        init: Box::new(typed_init),
                        span: def.span,
                    })
                }
                ast::Item::Static(def) => {
                    let declared_ty = self.resolve_type(&def.type_ann)?;
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.ty != declared_ty {
                        return Err(TypeError::new(
                            format!(
                                "type mismatch in static `{}`: expected `{:?}`, found `{:?}`",
                                def.name, declared_ty, typed_init.ty
                            ),
                            def.init.span,
                        ));
                    }

                    hir::Item::Static(hir::StaticDef {
                        name: def.name.clone(),
                        ty: declared_ty,
                        init: Box::new(typed_init),
                        span: def.span,
                    })
                }
                ast::Item::Function(def) => {
                    let typed_func = self.check_function(def)?;
                    hir::Item::Function(typed_func)
                }
                ast::Item::Extern(extern_item) => {
                    let hir_extern = match extern_item {
                        ast::ExternItem::Static(decl) => {
                            let ty = self.resolve_type(&decl.type_ann)?;
                            hir::ExternItem::Static(hir::ExternStaticDecl {
                                name: decl.name.clone(),
                                ty,
                                span: decl.span,
                            })
                        }
                        ast::ExternItem::Function(decl) => {
                            let params = decl
                                .params
                                .iter()
                                .map(|p| {
                                    let ty = self.resolve_type(&p.type_ann)?;
                                    Ok(hir::Param {
                                        name: p.name.clone(),
                                        ty,
                                        span: p.span,
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let return_ty = self.resolve_type(&decl.return_type_ann)?;
                            hir::ExternItem::Function(hir::ExternFunctionDecl {
                                name: decl.name.clone(),
                                params,
                                return_ty,
                                span: decl.span,
                            })
                        }
                    };
                    hir::Item::Extern(hir_extern)
                }
                ast::Item::Struct(_) => {
                    continue;
                }
            };

            typed_items.push(typed_item);
        }

        Ok(hir::Program { items: typed_items })
    }

    fn resolve_struct(&mut self, struct_def: &ast::StructDef) -> Result<(), TypeError> {
        let mut fields = Vec::new();
        let mut offset = 0;
        let mut max_align = 1;

        // Check field name uniqueness
        let mut field_names = HashSet::new();
        for field in &struct_def.fields {
            if !field_names.insert(&field.name) {
                return Err(TypeError::new(
                    format!(
                        "duplicate field `{}` in struct `{}`",
                        field.name, struct_def.name
                    ),
                    field.span,
                ));
            }
        }

        // Resolve field types and compute layout
        for field in &struct_def.fields {
            let field_ty = self.resolve_type(&field.type_ann)?;

            let field_align = field_ty.align as usize;
            max_align = max_align.max(field_align);

            if offset % field_align != 0 {
                offset += field_align - (offset % field_align);
            }

            fields.push(FieldInfo {
                name: field.name.clone(),
                ty: field_ty.clone(),
                offset,
            });

            offset += field_ty.size;
        }

        // Final size must be aligned to struct alignment
        let size = if offset % max_align != 0 {
            offset + max_align - (offset % max_align)
        } else {
            offset
        };

        let struct_type =
            Type::struct_type(struct_def.name.clone(), fields, size, max_align as u64);

        self.types.insert(struct_def.name.clone(), struct_type);

        Ok(())
    }

    fn check_function(&mut self, func: &ast::FunctionDef) -> Result<hir::FunctionDef, TypeError> {
        let return_ty = self.resolve_type(&func.return_type_ann)?;
        self.current_fn_return_ty = Some(return_ty.clone());
        self.push_scope();

        let mut hir_params = Vec::new();
        for param in &func.params {
            let param_ty = self.resolve_type(&param.type_ann)?;
            self.insert_local_var(param.name.clone(), param_ty.clone());
            hir_params.push(hir::Param {
                name: param.name.clone(),
                ty: param_ty,
                span: param.span,
            });
        }

        let typed_body = self.check_block(&func.body)?;

        // The `Never` type is compatible with any return type
        if typed_body.ty != return_ty && typed_body.ty.kind != TypeKind::Never {
            return Err(TypeError::new(
                format!(
                    "mismatched return type in function `{}`: expected `{:?}`, found `{:?}`",
                    func.name, return_ty, typed_body.ty
                ),
                func.body.span,
            ));
        }

        self.pop_scope();
        self.current_fn_return_ty = None;

        Ok(hir::FunctionDef {
            name: func.name.clone(),
            params: hir_params,
            return_ty,
            body: typed_body,
            span: func.span,
        })
    }

    fn check_block(&mut self, block: &ast::Block) -> Result<hir::Block, TypeError> {
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
                hir::StmtKind::Expr(expr)
                    if !is_last
                        && expr.ty.kind != TypeKind::Unit
                        && expr.ty.kind != TypeKind::Never =>
                {
                    return Err(TypeError::new(
                        format!(
                            "expected `;` after expression: expected `()`, found `{:?}`",
                            expr.ty
                        ),
                        expr.span,
                    ));
                }
                hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr)
                    if expr.ty.kind == TypeKind::Never =>
                {
                    has_never = true;
                }
                _ => {}
            }

            typed_stmts.push(typed_stmt);
        }

        let block_ty = match typed_stmts.last() {
            Some(hir::Stmt {
                kind: hir::StmtKind::Expr(expr),
                ..
            }) => expr.ty.clone(),
            Some(hir::Stmt {
                kind: hir::StmtKind::Semi(expr),
                ..
            }) if expr.ty.kind == TypeKind::Never => Type::never(),
            _ => Type::unit(),
        };

        self.pop_scope();

        Ok(hir::Block {
            stmts: typed_stmts,
            ty: block_ty,
            span: block.span,
        })
    }

    fn check_statement(&mut self, stmt: &ast::Stmt) -> Result<hir::Stmt, TypeError> {
        let kind = match &stmt.kind {
            ast::StmtKind::Let(let_stmt) => {
                let declared_ty = self.resolve_type(&let_stmt.type_ann)?;
                let typed_init = self.check_expression(&let_stmt.init)?;

                if typed_init.ty != declared_ty {
                    return Err(TypeError::new(
                        format!(
                            "type mismatch in let binding: expected `{:?}`, found `{:?}`",
                            declared_ty, typed_init.ty
                        ),
                        let_stmt.init.span,
                    ));
                }

                self.insert_local_var(let_stmt.name.clone(), declared_ty.clone());

                hir::StmtKind::Let(hir::Let {
                    name: let_stmt.name.clone(),
                    ty: declared_ty,
                    init: typed_init,
                    span: let_stmt.span,
                })
            }
            ast::StmtKind::Semi(expr) => {
                let typed_expr = self.check_expression(expr)?;
                hir::StmtKind::Semi(typed_expr)
            }
            ast::StmtKind::Expr(expr) => {
                let typed_expr = self.check_expression(expr)?;
                hir::StmtKind::Expr(typed_expr)
            }
        };

        Ok(hir::Stmt {
            kind,
            span: stmt.span,
        })
    }

    fn check_expression(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        match &expr.kind {
            ast::ExprKind::Literal(..) => Ok(Self::check_expr_literal(expr)),
            ast::ExprKind::Struct(..) => self.check_expr_struct(expr),
            ast::ExprKind::Ident(..) => self.check_expr_ident(expr),
            ast::ExprKind::Array(..) => self.check_expr_array(expr),
            ast::ExprKind::Repeat(..) => self.check_expr_repeat(expr),
            ast::ExprKind::Index(..) => self.check_expr_index(expr),
            ast::ExprKind::Call(..) => self.check_expr_call(expr),
            ast::ExprKind::Unary(..) => self.check_expr_unary(expr),
            ast::ExprKind::Binary(..) => self.check_expr_binary(expr),
            ast::ExprKind::Assign(..) => self.check_expr_assign(expr),
            ast::ExprKind::Return(..) => self.check_expr_return(expr),
            ast::ExprKind::Block(..) => self.check_expr_block(expr),
            ast::ExprKind::If(..) => self.check_expr_if(expr),
            ast::ExprKind::While(..) => self.check_expr_while(expr),
            ast::ExprKind::Break => Ok(hir::Expr {
                kind: hir::ExprKind::Break,
                ty: Type::never(),
                span: expr.span,
            }),
            ast::ExprKind::Continue => Ok(hir::Expr {
                kind: hir::ExprKind::Continue,
                ty: Type::never(),
                span: expr.span,
            }),
        }
    }

    fn check_expr_literal(expr: &ast::Expr) -> hir::Expr {
        let ast::ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        let (ty, kind) = match lit {
            ast::Literal::Integer(n) => (
                Type::i64(),
                hir::ExprKind::Literal(hir::Literal::Integer(*n)),
            ),
            ast::Literal::String(s) => (
                Type::slice(Type::u8()),
                hir::ExprKind::Literal(hir::Literal::String(s.clone())),
            ),
            ast::Literal::Bool(b) => (Type::bool(), hir::ExprKind::Literal(hir::Literal::Bool(*b))),
        };

        hir::Expr {
            kind,
            ty,
            span: expr.span,
        }
    }

    fn check_expr_struct(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Struct(struct_expr) = &expr.kind else {
            unreachable!()
        };

        let struct_ty = self
            .types
            .get(&struct_expr.name)
            .ok_or_else(|| {
                TypeError::new(
                    format!("undefined struct `{}`", struct_expr.name),
                    expr.span,
                )
            })?
            .clone();

        let TypeKind::Struct(struct_info) = &struct_ty.kind else {
            return Err(TypeError::new(
                format!("`{}` is not a struct type", struct_expr.name),
                expr.span,
            ));
        };

        // Build a map of provided fields
        let mut provided_fields = HashMap::new();
        for field_init in &struct_expr.fields {
            if let Some(prev) = provided_fields.insert(&field_init.name, field_init) {
                return Err(TypeError::new(
                    format!("duplicate field `{}` in struct literal", field_init.name),
                    prev.span,
                ));
            }
        }

        // Check all expected fields
        // Remove fields from the provided fields map, so any remaining are extra
        let mut typed_fields = Vec::new();
        for field_info in &struct_info.fields {
            let field_init = provided_fields.remove(&field_info.name).ok_or_else(|| {
                TypeError::new(
                    format!(
                        "missing field `{}` in struct literal for `{}`",
                        field_info.name, struct_expr.name
                    ),
                    expr.span,
                )
            })?;

            let typed_value = self.check_expression(&field_init.value)?;

            if typed_value.ty != field_info.ty {
                return Err(TypeError::new(
                    format!(
                        "field `{}` has wrong type: expected `{:?}`, found `{:?}`",
                        field_info.name, field_info.ty, typed_value.ty
                    ),
                    field_init.value.span,
                ));
            }

            typed_fields.push(hir::FieldInit {
                name: field_info.name.clone(),
                value: Box::new(typed_value),
                span: field_init.span,
            });
        }

        // Any remaining fields in the map are extra/unknown fields
        if let Some((name, field_init)) = provided_fields.into_iter().next() {
            return Err(TypeError::new(
                format!("struct `{}` has no field `{}`", struct_expr.name, name),
                field_init.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Struct(hir::StructExpr {
                name: struct_expr.name.clone(),
                fields: typed_fields,
                span: struct_expr.span,
            }),
            ty: struct_ty,
            span: expr.span,
        })
    }

    fn check_expr_ident(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Ident(name) = &expr.kind else {
            unreachable!()
        };

        let ty = self
            .lookup_var_type(name)
            .ok_or_else(|| TypeError::new(format!("undefined variable `{name}`"), expr.span))?
            .clone();

        Ok(hir::Expr {
            kind: hir::ExprKind::Ident(name.clone()),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_array(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Array(elems) = &expr.kind else {
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
                        "array element type mismatch: expected `{:?}`, found `{:?}`",
                        elem_ty, typed_elem.ty
                    ),
                    elem.span,
                ));
            }
            typed_elems.push(typed_elem);
        }

        let array_ty = Type::array(elem_ty, elems.len());
        Ok(hir::Expr {
            kind: hir::ExprKind::Array(typed_elems),
            ty: array_ty,
            span: expr.span,
        })
    }

    fn check_expr_repeat(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let typed_elem = self.check_expression(elem)?;
        let typed_count = self.check_expression(count)?;

        if typed_count.ty.kind != TypeKind::I64 {
            return Err(TypeError::new(
                format!("repeat count must be `i64`, found `{:?}`", typed_count.ty),
                count.span,
            ));
        }

        if let hir::ExprKind::Literal(hir::Literal::Integer(n)) = typed_count.kind {
            Ok(hir::Expr {
                kind: hir::ExprKind::Repeat(Box::new(typed_elem.clone()), Box::new(typed_count)),
                ty: Type::array(typed_elem.ty, n as usize),
                span: expr.span,
            })
        } else {
            Err(TypeError::new(
                "repeat count must be a constant integer".to_string(),
                count.span,
            ))
        }
    }

    fn check_expr_index(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Index(array, index) = &expr.kind else {
            unreachable!()
        };

        let typed_array = self.check_expression(array)?;
        let typed_index = self.check_expression(index)?;

        if typed_index.ty.kind != TypeKind::I64 {
            return Err(TypeError::new(
                format!("array index must be `i64`, found `{:?}`", typed_index.ty),
                index.span,
            ));
        }

        match typed_array.ty.kind {
            TypeKind::Array(ref elem_ty, _) | TypeKind::Slice(ref elem_ty) => Ok(hir::Expr {
                kind: hir::ExprKind::Index(Box::new(typed_array.clone()), Box::new(typed_index)),
                ty: *elem_ty.clone(),
                span: expr.span,
            }),
            _ => Err(TypeError::new(
                format!("cannot index into type `{:?}`", typed_array.ty),
                array.span,
            )),
        }
    }

    fn check_expr_call(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Call(call) = &expr.kind else {
            unreachable!()
        };

        let callee_name = if let ast::ExprKind::Ident(name) = &call.callee.kind {
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
                TypeError::new(format!("undefined function `{callee_name}`"), call.span)
            })?
            .clone();

        if call.args.len() != func_sig.params.len() {
            return Err(TypeError::new(
                format!(
                    "function `{callee_name}` expects {} arguments, got {}",
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
                        "argument type mismatch: expected `{:?}`, found `{:?}`",
                        param_ty, typed_arg.ty
                    ),
                    arg.span,
                ));
            }
            typed_args.push(typed_arg);
        }

        let typed_callee = hir::Expr {
            kind: hir::ExprKind::Ident(callee_name),
            ty: func_sig.return_ty.clone(),
            span: call.callee.span,
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Call(hir::Call {
                callee: Box::new(typed_callee),
                args: typed_args,
                span: call.span,
            }),
            ty: func_sig.return_ty,
            span: expr.span,
        })
    }

    fn check_expr_unary(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Unary(op, operand) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::UnaryOp::from(op);
        let typed_operand = self.check_expression(operand)?;

        let ty = match typed_op {
            hir::UnaryOp::Neg => {
                if typed_operand.ty.kind != TypeKind::I64 {
                    return Err(TypeError::new(
                        format!("cannot apply `-` to type `{:?}`", typed_operand.ty),
                        operand.span,
                    ));
                }
                Type::i64()
            }
            hir::UnaryOp::Not => match typed_operand.ty.kind {
                TypeKind::Bool => Type::bool(),
                TypeKind::I64 => Type::i64(),
                _ => {
                    return Err(TypeError::new(
                        format!("cannot apply `!` to type `{:?}`", typed_operand.ty),
                        operand.span,
                    ));
                }
            },
            hir::UnaryOp::Ref => {
                if let hir::ExprKind::Ident(name) = &typed_operand.kind
                    && self.constants.contains_key(name)
                {
                    return Err(TypeError::new(
                        format!("cannot take address of constant `{name}`"),
                        operand.span,
                    ));
                }
                Type::pointer(typed_operand.ty.clone())
            }
            hir::UnaryOp::Deref => match &typed_operand.ty.kind {
                TypeKind::Pointer(inner) => *inner.clone(),
                _ => {
                    return Err(TypeError::new(
                        format!(
                            "cannot dereference non-pointer type `{:?}`",
                            typed_operand.ty
                        ),
                        operand.span,
                    ));
                }
            },
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Unary(typed_op, Box::new(typed_operand)),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_binary(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Binary(op, left, right) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::BinaryOp::from(op);
        let typed_left = self.check_expression(left)?;
        let typed_right = self.check_expression(right)?;

        let ty = match typed_op {
            hir::BinaryOp::Add
            | hir::BinaryOp::Sub
            | hir::BinaryOp::Mul
            | hir::BinaryOp::Div
            | hir::BinaryOp::Rem
            | hir::BinaryOp::BitAnd
            | hir::BinaryOp::BitOr => {
                if typed_left.ty.kind != TypeKind::I64 || typed_right.ty.kind != TypeKind::I64 {
                    return Err(TypeError::new(
                        format!(
                            "arithmetic operator requires `i64` operands, found `{:?}` and `{:?}`",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::i64()
            }
            hir::BinaryOp::Lt | hir::BinaryOp::Le | hir::BinaryOp::Gt | hir::BinaryOp::Ge => {
                if typed_left.ty.kind != TypeKind::I64 || typed_right.ty.kind != TypeKind::I64 {
                    return Err(TypeError::new(
                        format!(
                            "comparison operator requires `i64` operands, found `{:?}` and `{:?}`",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::bool()
            }
            hir::BinaryOp::Eq | hir::BinaryOp::Ne => {
                if typed_left.ty != typed_right.ty {
                    return Err(TypeError::new(
                        format!(
                            "equality operator requires same types, found `{:?}` and `{:?}`",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::bool()
            }
            hir::BinaryOp::And | hir::BinaryOp::Or => {
                if typed_left.ty.kind != TypeKind::Bool || typed_right.ty.kind != TypeKind::Bool {
                    return Err(TypeError::new(
                        format!(
                            "logical operator requires `bool` operands, found `{:?}` and `{:?}`",
                            typed_left.ty, typed_right.ty
                        ),
                        expr.span,
                    ));
                }
                Type::bool()
            }
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Binary(typed_op, Box::new(typed_left), Box::new(typed_right)),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_assign(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let typed_lhs = self.check_expression(lhs)?;
        let typed_rhs = self.check_expression(rhs)?;

        if typed_lhs.ty != typed_rhs.ty {
            return Err(TypeError::new(
                format!(
                    "assignment type mismatch: expected `{:?}`, found `{:?}`",
                    typed_lhs.ty, typed_rhs.ty
                ),
                expr.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Assign(Box::new(typed_lhs), Box::new(typed_rhs)),
            ty: Type::unit(),
            span: expr.span,
        })
    }

    fn check_expr_return(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Return(val) = &expr.kind else {
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
                        "return type mismatch: expected `{:?}`, found `{:?}`",
                        return_ty, typed_val.ty
                    ),
                    v.span,
                ));
            }
            hir::ExprKind::Return(Some(Box::new(typed_val)))
        } else {
            if return_ty.kind != TypeKind::Unit {
                return Err(TypeError::new(
                    format!("expected return value of type `{return_ty:?}`"),
                    expr.span,
                ));
            }
            hir::ExprKind::Return(None)
        };

        Ok(hir::Expr {
            kind,
            ty: Type::never(),
            span: expr.span,
        })
    }

    fn check_expr_block(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Block(block) = &expr.kind else {
            unreachable!()
        };

        let typed_block = self.check_block(block)?;
        let block_ty = typed_block.ty.clone();

        Ok(hir::Expr {
            kind: hir::ExprKind::Block(typed_block),
            ty: block_ty,
            span: expr.span,
        })
    }

    fn check_expr_if(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&if_expr.cond)?;

        if typed_cond.ty.kind != TypeKind::Bool {
            return Err(TypeError::new(
                format!("if condition must be `bool`, found `{:?}`", typed_cond.ty),
                if_expr.cond.span,
            ));
        }

        let typed_then = self.check_block(&if_expr.then_body)?;

        let (ty, typed_else) = if let Some(else_expr) = &if_expr.else_body {
            let typed_else = self.check_expression(else_expr)?;

            let result_ty = match (&typed_then.ty.kind, &typed_else.ty.kind) {
                (TypeKind::Never, TypeKind::Never) => Type::never(),
                (_, TypeKind::Never) => typed_then.ty.clone(),
                (TypeKind::Never, _) => typed_else.ty.clone(),
                (then_ty, else_ty) if then_ty == else_ty => typed_then.ty.clone(),
                (then_ty, else_ty) => {
                    return Err(TypeError::new(
                        format!(
                            "if-else branches have different types: `{then_ty:?}` and `{else_ty:?}`",
                        ),
                        else_expr.span,
                    ));
                }
            };

            (result_ty, Some(Box::new(typed_else)))
        } else {
            // when there is no else-branch, the implicit else branch returns `Unit`,
            // so the entire `Expr` is always `Unit`
            // (even if the then-branch is `Never`)
            (Type::unit(), None)
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::If(hir::If {
                cond: Box::new(typed_cond),
                then_body: Box::new(typed_then),
                else_body: typed_else,
                span: if_expr.span,
            }),
            ty,
            span: expr.span,
        })
    }

    fn check_expr_while(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&while_expr.cond)?;

        if typed_cond.ty.kind != TypeKind::Bool {
            return Err(TypeError::new(
                format!(
                    "while condition must be `bool`, found `{:?}`",
                    typed_cond.ty
                ),
                while_expr.cond.span,
            ));
        }

        let typed_body = self.check_block(&while_expr.body)?;

        Ok(hir::Expr {
            kind: hir::ExprKind::While(hir::While {
                cond: Box::new(typed_cond),
                body: Box::new(typed_body),
                span: while_expr.span,
            }),
            ty: Type::unit(),
            span: expr.span,
        })
    }
}
