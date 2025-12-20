use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use crate::{
    ast, hir,
    scope::{Scope, ScopeObject},
    span::Span,
    types::{Field, TypeContext, TypeId, TypeKind},
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

pub struct TypeChecker<'a> {
    ctx: &'a mut TypeContext,
    scopes: Vec<Scope>,
    current_fn_return_type_id: Option<TypeId>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(ctx: &'a mut TypeContext) -> Self {
        Self {
            ctx,
            scopes: vec![Scope::new()],
            current_fn_return_type_id: None,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.scopes
            .pop()
            .expect("ICE: cannot pop scope, scopes stack is empty");
    }

    fn current_scope(&mut self) -> &mut Scope {
        self.scopes
            .last_mut()
            .expect("scope stack should not be empty")
    }

    fn global_scope(&mut self) -> &mut Scope {
        self.scopes
            .first_mut()
            .expect("scope stack should not be empty")
    }

    fn lookup(&self, name: &str) -> Option<&ScopeObject> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn resolve_type(&mut self, type_ann: &ast::TypeAnn) -> Result<TypeId, TypeError> {
        match &type_ann.kind {
            ast::TypeAnnKind::I64 => Ok(TypeId::I64),
            ast::TypeAnnKind::U8 => Ok(TypeId::U8),
            ast::TypeAnnKind::Bool => Ok(TypeId::BOOL),
            ast::TypeAnnKind::Unit => Ok(TypeId::UNIT),
            ast::TypeAnnKind::Never => Ok(TypeId::NEVER),

            ast::TypeAnnKind::Pointer(inner) => {
                let inner_type_id = self.resolve_type(inner)?;
                Ok(self.ctx.mk_pointer(inner_type_id))
            }

            ast::TypeAnnKind::Array(elem, count) => {
                let elem_type_id = self.resolve_type(elem)?;
                Ok(self.ctx.mk_array(elem_type_id, *count))
            }

            ast::TypeAnnKind::Slice(elem) => {
                let elem_type_id = self.resolve_type(elem)?;
                Ok(self.ctx.mk_slice(elem_type_id))
            }

            ast::TypeAnnKind::Named(name) => match self.lookup(name) {
                Some(ScopeObject::Type(type_id)) => Ok(*type_id),
                _ => Err(TypeError::new(
                    format!("undefined type `{name}`"),
                    type_ann.span,
                )),
            },
        }
    }

    pub fn check_program(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        self.collect_signatures(prog)?;
        self.check_items(prog)
    }

    fn register_struct_type(&mut self, def: &ast::StructDef) -> Result<(), TypeError> {
        let mut field_names = HashSet::new();
        let mut fields = Vec::new();
        let mut offset = 0;
        let mut max_align = 1;

        for field in &def.fields {
            if !field_names.insert(&field.name) {
                return Err(TypeError::new(
                    format!("duplicate field `{}` in struct `{}`", field.name, def.name),
                    field.span,
                ));
            }

            let field_type_id = self.resolve_type(&field.type_ann)?;
            let field_type = self.ctx.get(field_type_id);

            let field_align = field_type.align as usize;
            max_align = max_align.max(field_align);

            if offset % field_align != 0 {
                offset += field_align - (offset % field_align);
            }

            fields.push(Field {
                name: field.name.clone(),
                type_id: field_type_id,
                offset,
            });

            offset += field_type.size;
        }

        let type_id = self.ctx.mk_struct(fields);
        if self
            .global_scope()
            .insert(def.name.clone(), ScopeObject::Type(type_id))
            .is_some()
        {
            return Err(TypeError::new(
                format!("name `{}` already defined", def.name),
                def.span,
            ));
        }
        Ok(())
    }

    fn collect_signatures(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            match item {
                ast::Item::Const(def) => {
                    let type_id = self.resolve_type(&def.type_ann)?;
                    if self
                        .global_scope()
                        .insert(def.name.clone(), ScopeObject::Const(type_id))
                        .is_some()
                    {
                        return Err(TypeError::new(
                            format!("name `{}` already defined", def.name),
                            def.span,
                        ));
                    }
                }
                ast::Item::Static(def) => {
                    let type_id = self.resolve_type(&def.type_ann)?;
                    if self
                        .global_scope()
                        .insert(def.name.clone(), ScopeObject::Static(type_id))
                        .is_some()
                    {
                        return Err(TypeError::new(
                            format!("name `{}` already defined", def.name),
                            def.span,
                        ));
                    }
                }
                ast::Item::Function(def) => {
                    let params = def
                        .params
                        .iter()
                        .map(|param| self.resolve_type(&param.type_ann))
                        .collect::<Result<Vec<_>, _>>()?;
                    let return_ty = self.resolve_type(&def.return_type_ann)?;

                    if self
                        .global_scope()
                        .insert(def.name.clone(), ScopeObject::Function(params, return_ty))
                        .is_some()
                    {
                        return Err(TypeError::new(
                            format!("name `{}` already defined", def.name),
                            def.span,
                        ));
                    }
                }
                ast::Item::Extern(extern_item) => match extern_item {
                    ast::ExternItem::Static(decl) => {
                        let type_id = self.resolve_type(&decl.type_ann)?;
                        if self
                            .global_scope()
                            .insert(decl.name.clone(), ScopeObject::Static(type_id))
                            .is_some()
                        {
                            return Err(TypeError::new(
                                format!("name `{}` already defined", decl.name),
                                decl.span,
                            ));
                        }
                    }
                    ast::ExternItem::Function(decl) => {
                        let params = decl
                            .params
                            .iter()
                            .map(|param| self.resolve_type(&param.type_ann))
                            .collect::<Result<Vec<_>, _>>()?;
                        let return_ty = self.resolve_type(&decl.return_type_ann)?;

                        if self
                            .global_scope()
                            .insert(decl.name.clone(), ScopeObject::Function(params, return_ty))
                            .is_some()
                        {
                            return Err(TypeError::new(
                                format!("name `{}` already defined", decl.name),
                                decl.span,
                            ));
                        }
                    }
                },
                ast::Item::Struct(def) => self.register_struct_type(def)?,
            }
        }
        Ok(())
    }

    fn check_items(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        let mut typed_items = Vec::new();
        for item in &prog.items {
            let typed_item = match item {
                ast::Item::Const(def) => {
                    let declared_type_id = self.resolve_type(&def.type_ann)?;
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.type_id != declared_type_id {
                        return Err(TypeError::new(
                            format!(
                                "type mismatch in const `{}`: expected `{:?}`, found `{:?}`",
                                def.name, declared_type_id, typed_init.type_id
                            ),
                            def.init.span,
                        ));
                    }

                    hir::Item::Const(hir::ConstDef {
                        name: def.name.clone(),
                        type_id: declared_type_id,
                        init: Box::new(typed_init),
                        span: def.span,
                    })
                }
                ast::Item::Static(def) => {
                    let declared_type_id = self.resolve_type(&def.type_ann)?;
                    let typed_init = self.check_expression(&def.init)?;

                    if typed_init.type_id != declared_type_id {
                        return Err(TypeError::new(
                            format!(
                                "type mismatch in static `{}`: expected `{:?}`, found `{:?}`",
                                def.name, declared_type_id, typed_init.type_id
                            ),
                            def.init.span,
                        ));
                    }

                    hir::Item::Static(hir::StaticDef {
                        name: def.name.clone(),
                        type_id: declared_type_id,
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
                            let type_id = self.resolve_type(&decl.type_ann)?;
                            hir::ExternItem::Static(hir::ExternStaticDecl {
                                name: decl.name.clone(),
                                type_id,
                                span: decl.span,
                            })
                        }
                        ast::ExternItem::Function(decl) => {
                            let params = decl
                                .params
                                .iter()
                                .map(|p| {
                                    let type_id = self.resolve_type(&p.type_ann)?;
                                    Ok(hir::Param {
                                        name: p.name.clone(),
                                        type_id,
                                        span: p.span,
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let return_type_id = self.resolve_type(&decl.return_type_ann)?;
                            hir::ExternItem::Function(hir::ExternFunctionDecl {
                                name: decl.name.clone(),
                                params,
                                return_type_id,
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

    fn check_function(&mut self, func: &ast::FunctionDef) -> Result<hir::FunctionDef, TypeError> {
        let return_type_id = self.resolve_type(&func.return_type_ann)?;
        self.current_fn_return_type_id = Some(return_type_id);
        self.push_scope();

        let mut hir_params = Vec::new();
        for param in &func.params {
            let param_type_id = self.resolve_type(&param.type_ann)?;
            if self
                .current_scope()
                .insert(param.name.clone(), ScopeObject::Var(param_type_id))
                .is_some()
            {
                return Err(TypeError::new(
                    format!("parameter `{}` already defined", param.name),
                    param.span,
                ));
            }
            hir_params.push(hir::Param {
                name: param.name.clone(),
                type_id: param_type_id,
                span: param.span,
            });
        }

        let typed_body = self.check_block(&func.body)?;

        // The `Never` type is compatible with any return type
        if typed_body.type_id != return_type_id && typed_body.type_id != TypeId::NEVER {
            return Err(TypeError::new(
                format!(
                    "mismatched return type in function `{}`: expected `{:?}`, found `{:?}`",
                    func.name, return_type_id, typed_body.type_id
                ),
                func.body.span,
            ));
        }

        self.pop_scope();
        self.current_fn_return_type_id = None;

        Ok(hir::FunctionDef {
            name: func.name.clone(),
            params: hir_params,
            return_type_id,
            body: typed_body,
            span: func.span,
        })
    }

    fn check_block(&mut self, block: &ast::Block) -> Result<hir::Block, TypeError> {
        self.scopes.push(Scope::new());

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
                        && expr.type_id != TypeId::UNIT
                        && expr.type_id != TypeId::NEVER =>
                {
                    return Err(TypeError::new(
                        format!(
                            "expected `;` after expression: expected `()`, found `{:?}`",
                            expr.type_id
                        ),
                        expr.span,
                    ));
                }
                hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr)
                    if expr.type_id == TypeId::NEVER =>
                {
                    has_never = true;
                }
                _ => {}
            }

            typed_stmts.push(typed_stmt);
        }

        let block_type_id = match typed_stmts.last() {
            Some(hir::Stmt {
                kind: hir::StmtKind::Expr(expr),
                ..
            }) => expr.type_id,
            Some(hir::Stmt {
                kind: hir::StmtKind::Semi(expr),
                ..
            }) if expr.type_id == TypeId::NEVER => TypeId::NEVER,
            _ => TypeId::UNIT,
        };

        self.pop_scope();

        Ok(hir::Block {
            stmts: typed_stmts,
            type_id: block_type_id,
            span: block.span,
        })
    }

    fn check_statement(&mut self, stmt: &ast::Stmt) -> Result<hir::Stmt, TypeError> {
        let kind = match &stmt.kind {
            ast::StmtKind::Let(let_stmt) => {
                let declared_type_id = self.resolve_type(&let_stmt.type_ann)?;
                let typed_init = self.check_expression(&let_stmt.init)?;

                if typed_init.type_id != declared_type_id {
                    return Err(TypeError::new(
                        format!(
                            "type mismatch in let binding: expected `{:?}`, found `{:?}`",
                            declared_type_id, typed_init.type_id
                        ),
                        let_stmt.init.span,
                    ));
                }

                if self
                    .current_scope()
                    .insert(let_stmt.name.clone(), ScopeObject::Var(declared_type_id))
                    .is_some()
                {
                    return Err(TypeError::new(
                        format!("variable `{}` already defined", let_stmt.name),
                        let_stmt.span,
                    ));
                }

                hir::StmtKind::Let(hir::Let {
                    name: let_stmt.name.clone(),
                    type_id: declared_type_id,
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
            ast::ExprKind::Literal(..) => Ok(self.check_expr_literal(expr)),
            ast::ExprKind::Struct(..) => self.check_expr_struct(expr),
            ast::ExprKind::Ident(..) => self.check_expr_ident(expr),
            ast::ExprKind::Array(..) => self.check_expr_array(expr),
            ast::ExprKind::Repeat(..) => self.check_expr_repeat(expr),
            ast::ExprKind::Field(..) => self.check_expr_field(expr),
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
                type_id: TypeId::NEVER,
                span: expr.span,
            }),
            ast::ExprKind::Continue => Ok(hir::Expr {
                kind: hir::ExprKind::Continue,
                type_id: TypeId::NEVER,
                span: expr.span,
            }),
        }
    }

    fn check_expr_literal(&mut self, expr: &ast::Expr) -> hir::Expr {
        let ast::ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        let (ty, kind) = match lit {
            ast::Literal::Integer(n) => (
                TypeId::I64,
                hir::ExprKind::Literal(hir::Literal::Integer(*n)),
            ),
            ast::Literal::String(s) => (
                self.ctx.mk_slice(TypeId::U8),
                hir::ExprKind::Literal(hir::Literal::String(s.clone())),
            ),
            ast::Literal::Bool(b) => (TypeId::BOOL, hir::ExprKind::Literal(hir::Literal::Bool(*b))),
        };

        hir::Expr {
            kind,
            type_id: ty,
            span: expr.span,
        }
    }

    fn check_expr_struct(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Struct(struct_expr) = &expr.kind else {
            unreachable!()
        };

        let struct_type_id = match self.lookup(&struct_expr.name) {
            Some(ScopeObject::Type(type_id)) => *type_id,
            _ => {
                return Err(TypeError::new(
                    format!("undefined struct `{}`", struct_expr.name),
                    expr.span,
                ));
            }
        };

        let expected_fields = match &self.ctx.get(struct_type_id).kind {
            TypeKind::Struct(fields) => fields.clone(),
            _ => {
                return Err(TypeError::new(
                    format!("`{}` is not a struct", struct_expr.name),
                    expr.span,
                ));
            }
        };

        let mut provided_fields = HashMap::new();
        for field_init in &struct_expr.fields {
            if let Some(prev) = provided_fields.insert(&field_init.name, field_init) {
                return Err(TypeError::new(
                    format!("duplicate field `{}` in struct literal", field_init.name),
                    prev.span,
                ));
            }
        }

        let mut typed_fields = Vec::new();
        for field in &expected_fields {
            let field_init = provided_fields.remove(&field.name).ok_or_else(|| {
                TypeError::new(
                    format!(
                        "missing field `{}` in struct literal for `{}`",
                        field.name, struct_expr.name
                    ),
                    expr.span,
                )
            })?;

            let typed_value = self.check_expression(&field_init.value)?;

            if typed_value.type_id != field.type_id {
                return Err(TypeError::new(
                    format!(
                        "field `{}` has wrong type: expected `{:?}`, found `{:?}`",
                        field.name, field.type_id, typed_value.type_id
                    ),
                    field_init.value.span,
                ));
            }

            typed_fields.push(hir::FieldInit {
                name: field.name.clone(),
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
            type_id: struct_type_id,
            span: expr.span,
        })
    }

    fn check_expr_ident(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Ident(name) = &expr.kind else {
            unreachable!()
        };

        let type_id = match self.lookup(name) {
            Some(
                ScopeObject::Var(type_id)
                | ScopeObject::Const(type_id)
                | ScopeObject::Static(type_id),
            ) => *type_id,
            _ => {
                return Err(TypeError::new(
                    format!("undefined variable `{name}`"),
                    expr.span,
                ));
            }
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Ident(name.clone()),
            type_id,
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
        let elem_type_id = first_elem.type_id;
        typed_elems.push(first_elem);

        for elem in &elems[1..] {
            let typed_elem = self.check_expression(elem)?;
            if typed_elem.type_id != elem_type_id {
                return Err(TypeError::new(
                    format!(
                        "array element type mismatch: expected `{:?}`, found `{:?}`",
                        elem_type_id, typed_elem.type_id
                    ),
                    elem.span,
                ));
            }
            typed_elems.push(typed_elem);
        }

        let type_id = self.ctx.mk_array(elem_type_id, elems.len());
        Ok(hir::Expr {
            kind: hir::ExprKind::Array(typed_elems),
            type_id,
            span: expr.span,
        })
    }

    fn check_expr_repeat(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let typed_elem = self.check_expression(elem)?;
        let typed_count = self.check_expression(count)?;

        if typed_count.type_id != TypeId::I64 {
            return Err(TypeError::new(
                format!(
                    "repeat count must be `i64`, found `{:?}`",
                    typed_count.type_id
                ),
                count.span,
            ));
        }

        if let hir::ExprKind::Literal(hir::Literal::Integer(n)) = typed_count.kind {
            Ok(hir::Expr {
                kind: hir::ExprKind::Repeat(Box::new(typed_elem.clone()), Box::new(typed_count)),
                type_id: self.ctx.mk_array(typed_elem.type_id, n as usize),
                span: expr.span,
            })
        } else {
            Err(TypeError::new(
                "repeat count must be a constant integer".to_string(),
                count.span,
            ))
        }
    }

    fn check_expr_field(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Field(base, field_name) = &expr.kind else {
            unreachable!()
        };

        let typed_base = self.check_expression(base)?;

        let TypeKind::Struct(fields) = &self.ctx.get(typed_base.type_id).kind else {
            return Err(TypeError::new(
                "cannot access field on non-struct type".to_string(),
                base.span,
            ));
        };

        let field_info = fields
            .iter()
            .find(|field| &field.name == field_name)
            .ok_or_else(|| {
                TypeError::new(format!("no field `{field_name}` on struct"), expr.span)
            })?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Field(Box::new(typed_base), field_name.clone()),
            type_id: field_info.type_id,
            span: expr.span,
        })
    }

    fn check_expr_index(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Index(array, index) = &expr.kind else {
            unreachable!()
        };

        let typed_array = self.check_expression(array)?;
        let typed_index = self.check_expression(index)?;

        if typed_index.type_id != TypeId::I64 {
            return Err(TypeError::new(
                format!(
                    "array index must be `i64`, found `{:?}`",
                    typed_index.type_id
                ),
                index.span,
            ));
        }

        let (TypeKind::Slice(elem_type_id) | TypeKind::Array(elem_type_id, _)) =
            self.ctx.get(typed_array.type_id).kind
        else {
            return Err(TypeError::new(
                format!("cannot index into type `{:?}`", typed_array.type_id),
                array.span,
            ));
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Index(Box::new(typed_array.clone()), Box::new(typed_index)),
            type_id: elem_type_id,
            span: expr.span,
        })
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

        let (params, return_ty) = match self.lookup(&callee_name) {
            Some(ScopeObject::Function(params, return_ty)) => (params.clone(), *return_ty),
            _ => {
                return Err(TypeError::new(
                    format!("undefined function `{callee_name}`"),
                    call.span,
                ));
            }
        };

        if call.args.len() != params.len() {
            return Err(TypeError::new(
                format!(
                    "function `{callee_name}` expects {} arguments, got {}",
                    params.len(),
                    call.args.len()
                ),
                call.span,
            ));
        }

        let mut typed_args = Vec::new();
        for (arg, param_type_id) in call.args.iter().zip(params) {
            let typed_arg = self.check_expression(arg)?;
            if typed_arg.type_id != param_type_id {
                return Err(TypeError::new(
                    format!(
                        "argument type mismatch: expected `{:?}`, found `{:?}`",
                        param_type_id, typed_arg.type_id
                    ),
                    arg.span,
                ));
            }
            typed_args.push(typed_arg);
        }

        let typed_callee = hir::Expr {
            kind: hir::ExprKind::Ident(callee_name),
            type_id: return_ty,
            span: call.callee.span,
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Call(hir::Call {
                callee: Box::new(typed_callee),
                args: typed_args,
                span: call.span,
            }),
            type_id: return_ty,
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
                if typed_operand.type_id != TypeId::I64 {
                    return Err(TypeError::new(
                        format!("cannot apply `-` to type `{:?}`", typed_operand.type_id),
                        operand.span,
                    ));
                }
                TypeId::I64
            }
            hir::UnaryOp::Not => match typed_operand.type_id {
                TypeId::BOOL => TypeId::BOOL,
                TypeId::I64 => TypeId::I64,
                _ => {
                    return Err(TypeError::new(
                        format!("cannot apply `!` to type `{:?}`", typed_operand.type_id),
                        operand.span,
                    ));
                }
            },
            hir::UnaryOp::Ref => {
                if let hir::ExprKind::Ident(name) = &typed_operand.kind
                    && matches!(self.lookup(name), Some(ScopeObject::Const { .. }))
                {
                    return Err(TypeError::new(
                        format!("cannot take address of constant `{name}`"),
                        operand.span,
                    ));
                }
                self.ctx.mk_pointer(typed_operand.type_id)
            }
            hir::UnaryOp::Deref => match self.ctx.get(typed_operand.type_id).kind {
                TypeKind::Pointer(elem) => elem,
                _ => {
                    return Err(TypeError::new(
                        format!(
                            "cannot dereference non-pointer type `{:?}`",
                            typed_operand.type_id
                        ),
                        operand.span,
                    ));
                }
            },
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Unary(typed_op, Box::new(typed_operand)),
            type_id: ty,
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
                if typed_left.type_id != TypeId::I64 || typed_right.type_id != TypeId::I64 {
                    return Err(TypeError::new(
                        format!(
                            "arithmetic operator requires `i64` operands, found `{:?}` and `{:?}`",
                            typed_left.type_id, typed_right.type_id
                        ),
                        expr.span,
                    ));
                }
                TypeId::I64
            }
            hir::BinaryOp::Lt | hir::BinaryOp::Le | hir::BinaryOp::Gt | hir::BinaryOp::Ge => {
                if typed_left.type_id != TypeId::I64 || typed_right.type_id != TypeId::I64 {
                    return Err(TypeError::new(
                        format!(
                            "comparison operator requires `i64` operands, found `{:?}` and `{:?}`",
                            typed_left.type_id, typed_right.type_id
                        ),
                        expr.span,
                    ));
                }
                TypeId::BOOL
            }
            hir::BinaryOp::Eq | hir::BinaryOp::Ne => {
                if typed_left.type_id != typed_right.type_id {
                    return Err(TypeError::new(
                        format!(
                            "equality operator requires same types, found `{:?}` and `{:?}`",
                            typed_left.type_id, typed_right.type_id
                        ),
                        expr.span,
                    ));
                }
                TypeId::BOOL
            }
            hir::BinaryOp::And | hir::BinaryOp::Or => {
                if typed_left.type_id != TypeId::BOOL || typed_right.type_id != TypeId::BOOL {
                    return Err(TypeError::new(
                        format!(
                            "logical operator requires `bool` operands, found `{:?}` and `{:?}`",
                            typed_left.type_id, typed_right.type_id
                        ),
                        expr.span,
                    ));
                }
                TypeId::BOOL
            }
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Binary(typed_op, Box::new(typed_left), Box::new(typed_right)),
            type_id: ty,
            span: expr.span,
        })
    }

    fn check_expr_assign(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let typed_lhs = self.check_expression(lhs)?;
        let typed_rhs = self.check_expression(rhs)?;

        if typed_lhs.type_id != typed_rhs.type_id {
            return Err(TypeError::new(
                format!(
                    "assignment type mismatch: expected `{:?}`, found `{:?}`",
                    typed_lhs.type_id, typed_rhs.type_id
                ),
                expr.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Assign(Box::new(typed_lhs), Box::new(typed_rhs)),
            type_id: TypeId::UNIT,
            span: expr.span,
        })
    }

    fn check_expr_return(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Return(val) = &expr.kind else {
            unreachable!()
        };

        let return_type_id = self
            .current_fn_return_type_id
            .ok_or_else(|| TypeError::new("return outside of function".to_string(), expr.span))?;

        let kind = if let Some(v) = val {
            let typed_val = self.check_expression(v)?;
            if typed_val.type_id != return_type_id {
                return Err(TypeError::new(
                    format!(
                        "return type mismatch: expected `{:?}`, found `{:?}`",
                        return_type_id, typed_val.type_id
                    ),
                    v.span,
                ));
            }
            hir::ExprKind::Return(Some(Box::new(typed_val)))
        } else {
            if return_type_id != TypeId::UNIT {
                return Err(TypeError::new(
                    format!("expected return value of type `{return_type_id:?}`"),
                    expr.span,
                ));
            }
            hir::ExprKind::Return(None)
        };

        Ok(hir::Expr {
            kind,
            type_id: TypeId::NEVER,
            span: expr.span,
        })
    }

    fn check_expr_block(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Block(block) = &expr.kind else {
            unreachable!()
        };

        let typed_block = self.check_block(block)?;
        let type_id = typed_block.type_id;

        Ok(hir::Expr {
            kind: hir::ExprKind::Block(typed_block),
            type_id,
            span: expr.span,
        })
    }

    fn check_expr_if(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&if_expr.cond)?;

        if typed_cond.type_id != TypeId::BOOL {
            return Err(TypeError::new(
                format!(
                    "if condition must be `bool`, found `{:?}`",
                    typed_cond.type_id
                ),
                if_expr.cond.span,
            ));
        }

        let typed_then = self.check_block(&if_expr.then_body)?;

        let (ty, typed_else) = if let Some(else_expr) = &if_expr.else_body {
            let typed_else = self.check_expression(else_expr)?;

            let result_ty = match (typed_then.type_id, typed_else.type_id) {
                (TypeId::NEVER, TypeId::NEVER) => TypeId::NEVER,
                (_, TypeId::NEVER) => typed_then.type_id,
                (TypeId::NEVER, _) => typed_else.type_id,
                (then_ty, else_ty) if then_ty == else_ty => typed_then.type_id,
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
            (TypeId::UNIT, None)
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::If(hir::If {
                cond: Box::new(typed_cond),
                then_body: Box::new(typed_then),
                else_body: typed_else,
                span: if_expr.span,
            }),
            type_id: ty,
            span: expr.span,
        })
    }

    fn check_expr_while(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&while_expr.cond)?;

        if typed_cond.type_id != TypeId::BOOL {
            return Err(TypeError::new(
                format!(
                    "while condition must be `bool`, found `{:?}`",
                    typed_cond.type_id
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
            type_id: TypeId::UNIT,
            span: expr.span,
        })
    }
}
