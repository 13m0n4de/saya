use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt, mem,
    path::PathBuf,
    rc::Rc,
};

use crate::{
    ast, hir,
    lexer::Lexer,
    parser::Parser,
    scope::{
        Const, ExternFunction, ExternStatic, Function, Scope, ScopeKind, ScopeObject, Scopes,
        Static, Struct, TypeAlias,
    },
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

type StructLayout = (Vec<(String, usize)>, usize, usize);

pub struct TypeChecker<'a> {
    pub scopes: Scopes,
    types: &'a mut TypeContext,
    namespace: Option<String>,
    td_paths: HashMap<String, String>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        types: &'a mut TypeContext,
        namespace: Option<String>,
        td_paths: HashMap<String, String>,
    ) -> Self {
        let mut scopes = Scopes::new();
        scopes.push(Scope {
            kind: ScopeKind::Module,
            objects: HashMap::new(),
        });

        Self {
            scopes,
            types,
            namespace,
            td_paths,
        }
    }

    pub fn check(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        self.load_uses(prog)?;
        self.scan_declarations(prog)?;
        self.lower_items(prog)
    }

    fn make_ident(&self, path: &ast::Path) -> String {
        if let Some(ns) = &self.namespace
            && path.segments.len() == 1
        {
            format!("{ns}::{path}")
        } else {
            path.to_string()
        }
    }

    fn eval_const_expr(&mut self, expr: &hir::Expr) -> Result<hir::ConstVal, TypeError> {
        match &expr.kind {
            hir::ExprKind::Literal(lit) => Ok(match lit {
                hir::Literal::Integer(n) => {
                    hir::ConstVal::new(hir::ConstValKind::Integer(*n), expr.type_id)
                }
                hir::Literal::Float(n) => {
                    hir::ConstVal::new(hir::ConstValKind::Float(*n), expr.type_id)
                }
                hir::Literal::Bool(b) => {
                    hir::ConstVal::new(hir::ConstValKind::Bool(*b), expr.type_id)
                }
                hir::Literal::String(s) => {
                    hir::ConstVal::new(hir::ConstValKind::String(s.clone()), expr.type_id)
                }
                hir::Literal::CString(s) => {
                    hir::ConstVal::new(hir::ConstValKind::CString(s.clone()), expr.type_id)
                }
                hir::Literal::Null => hir::ConstVal::new(hir::ConstValKind::Null, expr.type_id),
            }),

            hir::ExprKind::Const(val) => Ok(val.clone()),
            hir::ExprKind::Place(hir::Place::Local(symbol) | hir::Place::Global(symbol)) => {
                match self.scopes.lookup(symbol) {
                    Some(ScopeObject::Const(Const::Resolved(val))) => Ok(val.clone()),
                    _ => Err(TypeError::new(
                        format!("Constant `{symbol}` not found"),
                        expr.span,
                    )),
                }
            }
            hir::ExprKind::Struct(struct_expr) => {
                let type_kind = self.types.get(expr.type_id).kind.clone();
                let TypeKind::Struct(_, fields) = type_kind else {
                    unreachable!()
                };
                let struct_fields = struct_expr.fields.clone();
                let mut values = Vec::with_capacity(fields.len());
                for field in &fields {
                    let field_init = struct_fields
                        .iter()
                        .find(|f| f.name == field.name)
                        .ok_or_else(|| {
                            TypeError::new(
                                format!("missing field `{}` in struct literal", field.name),
                                expr.span,
                            )
                        })?;
                    values.push(self.eval_const_expr(&field_init.value)?);
                }
                Ok(hir::ConstVal::new(
                    hir::ConstValKind::Struct(values),
                    expr.type_id,
                ))
            }
            hir::ExprKind::Array(elems) => {
                let mut values = Vec::with_capacity(elems.len());
                for elem in elems {
                    values.push(self.eval_const_expr(elem)?);
                }
                Ok(hir::ConstVal::new(
                    hir::ConstValKind::Array(values),
                    expr.type_id,
                ))
            }
            hir::ExprKind::Repeat(elem, count) => {
                let elem_val = self.eval_const_expr(elem)?;
                Ok(hir::ConstVal::new(
                    hir::ConstValKind::Repeat(Box::new(elem_val), *count),
                    expr.type_id,
                ))
            }
            hir::ExprKind::Unary(hir::UnaryOp::Neg, operand) => {
                let val = self.eval_const_expr(operand)?;
                match val.kind {
                    hir::ConstValKind::Integer(n) => Ok(hir::ConstVal {
                        kind: hir::ConstValKind::Integer(-n),
                        type_id: expr.type_id,
                    }),
                    hir::ConstValKind::Float(n) => Ok(hir::ConstVal {
                        kind: hir::ConstValKind::Float(-n),
                        type_id: expr.type_id,
                    }),
                    _ => Err(TypeError::new(
                        "Cannot negate a non-numeric value".to_string(),
                        expr.span,
                    )),
                }
            }
            hir::ExprKind::Binary(op, left, right) => {
                let left_val = self.eval_const_expr(left)?;
                let right_val = self.eval_const_expr(right)?;

                let kind = match (left_val.kind, right_val.kind) {
                    (hir::ConstValKind::Integer(l), hir::ConstValKind::Integer(r)) => match op {
                        // Arithmetic operators
                        hir::BinaryOp::Add => hir::ConstValKind::Integer(l + r),
                        hir::BinaryOp::Sub => hir::ConstValKind::Integer(l - r),
                        hir::BinaryOp::Mul => hir::ConstValKind::Integer(l * r),
                        hir::BinaryOp::Div => hir::ConstValKind::Integer(l / r),
                        hir::BinaryOp::Rem => hir::ConstValKind::Integer(l % r),
                        // Bitwise operators
                        hir::BinaryOp::BitAnd => hir::ConstValKind::Integer(l & r),
                        hir::BinaryOp::BitOr => hir::ConstValKind::Integer(l | r),
                        // Comparison operators
                        hir::BinaryOp::Lt => hir::ConstValKind::Bool(l < r),
                        hir::BinaryOp::Le => hir::ConstValKind::Bool(l <= r),
                        hir::BinaryOp::Gt => hir::ConstValKind::Bool(l > r),
                        hir::BinaryOp::Ge => hir::ConstValKind::Bool(l >= r),
                        hir::BinaryOp::Eq => hir::ConstValKind::Bool(l == r),
                        hir::BinaryOp::Ne => hir::ConstValKind::Bool(l != r),
                        _ => {
                            return Err(TypeError::new(
                                "Invalid operator for integer operands".to_string(),
                                expr.span,
                            ));
                        }
                    },
                    (hir::ConstValKind::Float(l), hir::ConstValKind::Float(r)) => match op {
                        // Arithmetic operators
                        hir::BinaryOp::Add => hir::ConstValKind::Float(l + r),
                        hir::BinaryOp::Sub => hir::ConstValKind::Float(l - r),
                        hir::BinaryOp::Mul => hir::ConstValKind::Float(l * r),
                        hir::BinaryOp::Div => hir::ConstValKind::Float(l / r),
                        hir::BinaryOp::Rem => hir::ConstValKind::Float(l % r),
                        // Comparison perators
                        hir::BinaryOp::Lt => hir::ConstValKind::Bool(l < r),
                        hir::BinaryOp::Le => hir::ConstValKind::Bool(l <= r),
                        hir::BinaryOp::Gt => hir::ConstValKind::Bool(l > r),
                        hir::BinaryOp::Ge => hir::ConstValKind::Bool(l >= r),
                        hir::BinaryOp::Eq => hir::ConstValKind::Bool(l == r),
                        hir::BinaryOp::Ne => hir::ConstValKind::Bool(l != r),
                        _ => {
                            return Err(TypeError::new(
                                "Invalid operator for float operands".to_string(),
                                expr.span,
                            ));
                        }
                    },
                    (hir::ConstValKind::Bool(l), hir::ConstValKind::Bool(r)) => match op {
                        // Logical operators
                        hir::BinaryOp::And => hir::ConstValKind::Bool(l && r),
                        hir::BinaryOp::Or => hir::ConstValKind::Bool(l || r),
                        // Equality operators
                        hir::BinaryOp::Eq => hir::ConstValKind::Bool(l == r),
                        hir::BinaryOp::Ne => hir::ConstValKind::Bool(l != r),
                        _ => {
                            return Err(TypeError::new(
                                "Invalid operator for boolean operands".to_string(),
                                expr.span,
                            ));
                        }
                    },
                    _ => {
                        return Err(TypeError::new(
                            "Type mismatch in constant expression".to_string(),
                            expr.span,
                        ));
                    }
                };

                Ok(hir::ConstVal {
                    kind,
                    type_id: expr.type_id,
                })
            }
            _ => Err(TypeError::new(
                "Invalid constant expression".to_string(),
                expr.span,
            )),
        }
    }

    fn type_dimensions(&mut self, type_ann: &ast::TypeAnn) -> Result<(usize, usize), TypeError> {
        match &type_ann.kind {
            ast::TypeAnnKind::U8 | ast::TypeAnnKind::I8 | ast::TypeAnnKind::Bool => Ok((1, 1)),

            ast::TypeAnnKind::U16 | ast::TypeAnnKind::I16 => Ok((2, 2)),

            ast::TypeAnnKind::U32 | ast::TypeAnnKind::I32 | ast::TypeAnnKind::F32 => Ok((4, 4)),

            ast::TypeAnnKind::U64
            | ast::TypeAnnKind::I64
            | ast::TypeAnnKind::F64
            | ast::TypeAnnKind::Pointer(_)
            | ast::TypeAnnKind::Fn(_, _, _) => Ok((8, 8)),

            ast::TypeAnnKind::Unit | ast::TypeAnnKind::Never => Ok((0, 1)),

            ast::TypeAnnKind::Slice(_) => Ok((16, 8)),

            ast::TypeAnnKind::Array(elem, len_expr) => {
                let (elem_size, elem_align) = self.type_dimensions(elem)?;

                let typed_len = self.check_expression(len_expr, TypeId::I64)?;

                let evaluated_len = self.eval_const_expr(&typed_len)?;

                let hir::ConstValKind::Integer(len_val) = evaluated_len.kind else {
                    return Err(TypeError::new(
                        "Array length must be an integer".to_string(),
                        len_expr.span,
                    ));
                };

                if len_val <= 0 {
                    return Err(TypeError::new(
                        format!("Array length must be positive, found {len_val}"),
                        len_expr.span,
                    ));
                }

                let len = len_val as usize;
                Ok((elem_size * len, elem_align))
            }

            ast::TypeAnnKind::Path(path) => {
                let name = path.to_string();
                self.resolve_declaration(&name)?;

                match self.scopes.lookup(&name) {
                    Some(
                        ScopeObject::Struct(Struct::Resolved(type_id))
                        | ScopeObject::TypeAlias(TypeAlias::Resolved(type_id)),
                    ) => {
                        let t = self.types.get(*type_id);
                        Ok((t.size, t.align))
                    }
                    _ => Err(TypeError::new(
                        format!("undefined type `{path}`"),
                        type_ann.span,
                    )),
                }
            }
            ast::TypeAnnKind::Opaque => Err(TypeError::new(
                "opaque type cannot be used as a struct field".to_string(),
                type_ann.span,
            )),
        }
    }

    fn lower_type(&mut self, type_ann: &ast::TypeAnn) -> Result<TypeId, TypeError> {
        match &type_ann.kind {
            ast::TypeAnnKind::U8 => Ok(TypeId::U8),
            ast::TypeAnnKind::U16 => Ok(TypeId::U16),
            ast::TypeAnnKind::U32 => Ok(TypeId::U32),
            ast::TypeAnnKind::U64 => Ok(TypeId::U64),

            ast::TypeAnnKind::I8 => Ok(TypeId::I8),
            ast::TypeAnnKind::I16 => Ok(TypeId::I16),
            ast::TypeAnnKind::I32 => Ok(TypeId::I32),
            ast::TypeAnnKind::I64 => Ok(TypeId::I64),

            ast::TypeAnnKind::F32 => Ok(TypeId::F32),
            ast::TypeAnnKind::F64 => Ok(TypeId::F64),

            ast::TypeAnnKind::Bool => Ok(TypeId::Bool),

            ast::TypeAnnKind::Unit => Ok(TypeId::Unit),
            ast::TypeAnnKind::Never => Ok(TypeId::Never),
            ast::TypeAnnKind::Opaque => Ok(TypeId::Opaque),

            ast::TypeAnnKind::Pointer(inner) => {
                let inner_type_id = self.lower_type(inner)?;
                Ok(self.types.mk_pointer(inner_type_id))
            }

            ast::TypeAnnKind::Array(elem, len_expr) => {
                let elem_type_id = self.lower_type(elem)?;

                let typed_len = self.infer_expression(len_expr)?;

                if typed_len.type_id != TypeId::I64 {
                    return Err(TypeError::new(
                        format!(
                            "Array length must be `i64`, found `{}`",
                            self.types.type_name(typed_len.type_id)
                        ),
                        len_expr.span,
                    ));
                }

                let evaluated_len = self.eval_const_expr(&typed_len)?;

                let hir::ConstValKind::Integer(len_val) = evaluated_len.kind else {
                    return Err(TypeError::new(
                        "Array length must be an integer".to_string(),
                        len_expr.span,
                    ));
                };

                if len_val <= 0 {
                    return Err(TypeError::new(
                        format!("Array length must be positive, found {len_val}"),
                        len_expr.span,
                    ));
                }

                let len = len_val as usize;
                Ok(self.types.mk_array(elem_type_id, len))
            }

            ast::TypeAnnKind::Slice(elem) => {
                let elem_type_id = self.lower_type(elem)?;
                Ok(self.types.mk_slice(elem_type_id))
            }

            ast::TypeAnnKind::Path(path) => {
                let name = path.to_string();
                self.resolve_declaration(&name)?;

                match self.scopes.lookup(&name) {
                    Some(
                        ScopeObject::Struct(Struct::Resolved(type_id))
                        | ScopeObject::TypeAlias(TypeAlias::Resolved(type_id)),
                    ) => Ok(*type_id),
                    _ => Err(TypeError::new(
                        format!("undefined type `{path}`"),
                        type_ann.span,
                    )),
                }
            }

            ast::TypeAnnKind::Fn(params_type_ann, return_type_ann, is_variadic) => {
                let params_type_id = params_type_ann
                    .iter()
                    .map(|ann| self.lower_type(ann))
                    .collect::<Result<Vec<_>, _>>()?;
                let return_type_id = self.lower_type(return_type_ann)?;
                Ok(self
                    .types
                    .mk_fn(params_type_id, return_type_id, *is_variadic))
            }
        }
    }

    fn struct_layout(&mut self, def: &ast::StructDef) -> Result<StructLayout, TypeError> {
        let mut field_layouts = Vec::new();
        let mut offset = 0;
        let mut max_align = 1;

        for field in &def.fields {
            let (field_size, field_align) = self.type_dimensions(&field.type_ann)?;
            max_align = max_align.max(field_align);

            if offset % field_align != 0 {
                offset += field_align - (offset % field_align);
            }

            field_layouts.push((field.name.clone(), offset));
            offset += field_size;
        }

        let size = if offset % max_align != 0 {
            offset + max_align - (offset % max_align)
        } else {
            offset
        };

        Ok((field_layouts, size, max_align))
    }

    fn load_uses(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            let ast::ItemKind::Use(use_item) = &item.kind else {
                continue;
            };

            let module_name = use_item.path.to_string();
            let file_path: PathBuf = self
                .td_paths
                .get(&module_name)
                .map(PathBuf::from)
                .ok_or_else(|| {
                    TypeError::new(
                        format!("module `{module_name}` not found, use '-M {module_name}=<path>' to specify its typedef file"),
                        use_item.span,
                    )
                })?;

            let code = std::fs::read_to_string(&file_path).map_err(|e| {
                TypeError::new(
                    format!("failed to read module file `{}`: {e}", file_path.display()),
                    use_item.span,
                )
            })?;

            let lexer = Lexer::new(&code);
            let mut parser = Parser::new(lexer).map_err(|e| {
                TypeError::new(
                    format!("failed to parse module `{}`: {}", use_item.name, e),
                    use_item.span,
                )
            })?;

            let program = parser.parse().map_err(|e| {
                TypeError::new(
                    format!("failed to parse module `{}`: {}", use_item.name, e),
                    use_item.span,
                )
            })?;

            let mut checker = TypeChecker::new(self.types, None, HashMap::new());
            checker.check(&program).map_err(|e| {
                TypeError::new(
                    format!("failed to type check module `{}`: {}", use_item.name, e),
                    use_item.span,
                )
            })?;

            let objects = mem::take(&mut checker.scopes.first_mut().objects);
            self.scopes.first_mut().extend(objects);
        }
        Ok(())
    }

    fn scan_declarations(&mut self, prog: &ast::Program) -> Result<(), TypeError> {
        for item in &prog.items {
            let (name, obj, span) = match &item.kind {
                ast::ItemKind::Const(def) => (
                    def.path.to_string(),
                    ScopeObject::Const(Const::Unresolved(Rc::new(def.clone()))),
                    def.span,
                ),
                ast::ItemKind::Static(def) => (
                    def.path.to_string(),
                    ScopeObject::Static(Static::Unresolved(Rc::new(item.clone()))),
                    def.span,
                ),
                ast::ItemKind::Function(def) => (
                    def.path.to_string(),
                    ScopeObject::Function(Function::Unresolved(Rc::new(item.clone()))),
                    def.span,
                ),
                ast::ItemKind::Struct(def) => (
                    def.path.to_string(),
                    ScopeObject::Struct(Struct::Unresolved(Rc::new(def.clone()))),
                    def.span,
                ),
                ast::ItemKind::TypeAlias(def) => (
                    def.path.to_string(),
                    ScopeObject::TypeAlias(TypeAlias::Unresolved(Rc::new(def.clone()))),
                    def.span,
                ),
                ast::ItemKind::Extern(ast::ExternItem::Static(decl)) => (
                    decl.path.to_string(),
                    ScopeObject::ExternStatic(ExternStatic::Unresolved(Rc::new(item.clone()))),
                    decl.span,
                ),
                ast::ItemKind::Extern(ast::ExternItem::Function(decl)) => (
                    decl.path.to_string(),
                    ScopeObject::ExternFunction(ExternFunction::Unresolved(Rc::new(item.clone()))),
                    decl.span,
                ),
                ast::ItemKind::Use(_) => continue,
            };

            if self.scopes.first_mut().insert(name.clone(), obj).is_some() {
                return Err(TypeError::new(
                    format!("name `{name}` already defined"),
                    span,
                ));
            }
        }

        Ok(())
    }

    fn resolve_declaration(&mut self, name: &str) -> Result<(), TypeError> {
        let Some(obj) = self.scopes.lookup(name).cloned() else {
            return Ok(());
        };
        match &obj {
            ScopeObject::Const(_) => self.resolve_const_decl(name, obj),
            ScopeObject::Static(_) => self.resolve_static_decl(name, obj),
            ScopeObject::Function(_) => self.resolve_function_decl(name, obj),
            ScopeObject::Struct(_) => self.resolve_struct_decl(name, obj),
            ScopeObject::TypeAlias(_) => self.resolve_type_alias(name, obj),
            ScopeObject::ExternStatic(_) => self.resolve_extern_static_decl(name, obj),
            ScopeObject::ExternFunction(_) => self.resolve_extern_function_decl(name, obj),
            ScopeObject::Var(_) => Ok(()),
        }
    }
    fn resolve_const_decl(&mut self, name: &str, obj: ScopeObject) -> Result<(), TypeError> {
        let ScopeObject::Const(decl) = obj else {
            unreachable!()
        };

        match decl {
            Const::Unresolved(def) => {
                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Const(Const::Resolving(def.clone())),
                );

                let type_id = self.lower_type(&def.type_ann)?;
                let typed_init = self.check_expression(&def.init, type_id)?;

                let value = self.eval_const_expr(&typed_init)?;

                self.scopes
                    .first_mut()
                    .insert(name.to_string(), ScopeObject::Const(Const::Resolved(value)));

                Ok(())
            }
            Const::Resolving(def) => Err(TypeError::new(
                format!("circular dependency for const `{name}`"),
                def.span,
            )),
            Const::Resolved(_) => Ok(()),
        }
    }

    fn resolve_static_decl(&mut self, name: &str, obj: ScopeObject) -> Result<(), TypeError> {
        let ScopeObject::Static(decl) = obj else {
            unreachable!()
        };

        match decl {
            Static::Unresolved(item) => {
                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Static(Static::Resolving(item.clone())),
                );

                let ast::ItemKind::Static(def) = &item.kind else {
                    unreachable!()
                };

                let type_id = self.lower_type(&def.type_ann)?;
                let typed_init = self.check_expression(&def.init, type_id)?;

                let value = self.eval_const_expr(&typed_init)?;
                let ident = self.make_ident(&def.path);
                let mut symbol_override = None;
                for attr in &item.attrs {
                    match &attr.kind {
                        ast::AttrKind::Symbol(s) => {
                            if symbol_override.replace(s.clone()).is_some() {
                                return Err(TypeError::new("duplicate @symbol".into(), attr.span));
                            }
                        }
                    }
                }
                let symbol = symbol_override.unwrap_or_else(|| ident.replace("::", "."));

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Static(Static::Resolved { value, symbol }),
                );

                Ok(())
            }
            Static::Resolving(item) => {
                let ast::ItemKind::Static(def) = &item.kind else {
                    unreachable!()
                };
                Err(TypeError::new(
                    format!("circular dependency for static `{name}`"),
                    def.span,
                ))
            }
            Static::Resolved { .. } => Ok(()),
        }
    }

    fn resolve_function_decl(&mut self, name: &str, obj: ScopeObject) -> Result<(), TypeError> {
        let ScopeObject::Function(decl) = obj else {
            unreachable!()
        };

        match decl {
            Function::Unresolved(item) => {
                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Function(Function::Resolving(item.clone())),
                );

                let ast::ItemKind::Function(def) = &item.kind else {
                    unreachable!()
                };

                let params = def
                    .params
                    .iter()
                    .map(|p| self.lower_type(&p.type_ann))
                    .collect::<Result<Vec<_>, _>>()?;
                let return_ty = self.lower_type(&def.return_type_ann)?;

                let ident = self.make_ident(&def.path);
                let mut symbol_override = None;
                for attr in &item.attrs {
                    match &attr.kind {
                        ast::AttrKind::Symbol(s) => {
                            if symbol_override.replace(s.clone()).is_some() {
                                return Err(TypeError::new("duplicate @symbol".into(), attr.span));
                            }
                        }
                    }
                }
                let symbol = symbol_override.unwrap_or_else(|| ident.replace("::", "."));

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Function(Function::Resolved {
                        params,
                        ret: return_ty,
                        symbol,
                    }),
                );

                Ok(())
            }
            Function::Resolving(item) => {
                let ast::ItemKind::Function(def) = &item.kind else {
                    unreachable!()
                };
                Err(TypeError::new(
                    format!("circular dependency for function `{name}`"),
                    def.span,
                ))
            }
            Function::Resolved { .. } => Ok(()),
        }
    }

    fn resolve_struct_decl(&mut self, name: &str, obj: ScopeObject) -> Result<(), TypeError> {
        let ScopeObject::Struct(decl) = obj else {
            unreachable!()
        };

        match decl {
            Struct::Unresolved(def) => {
                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Struct(Struct::Resolving(def.clone())),
                );

                let mut field_names = HashSet::new();
                for field in &def.fields {
                    if !field_names.insert(&field.name) {
                        return Err(TypeError::new(
                            format!("duplicate field `{}` in struct `{name}`", field.name),
                            field.span,
                        ));
                    }
                }

                let (field_layouts, size, align) = self.struct_layout(&def)?;

                let struct_type_id = self.types.mk_empty_struct();

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::Struct(Struct::Resolved(struct_type_id)),
                );

                let fields = def
                    .fields
                    .iter()
                    .zip(field_layouts.iter())
                    .map(|(field, (field_name, offset))| {
                        let type_id = self.lower_type(&field.type_ann)?;
                        Ok(Field {
                            name: field_name.clone(),
                            type_id,
                            offset: *offset,
                        })
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?;

                self.types.set_struct(
                    struct_type_id,
                    self.make_ident(&def.path),
                    fields,
                    size,
                    align,
                );

                Ok(())
            }
            Struct::Resolving(def) => Err(TypeError::new(
                format!("circular dependency for struct `{name}`"),
                def.span,
            )),
            Struct::Resolved(_) => Ok(()),
        }
    }

    fn resolve_type_alias(&mut self, name: &str, obj: ScopeObject) -> Result<(), TypeError> {
        let ScopeObject::TypeAlias(decl) = obj else {
            unreachable!()
        };

        match decl {
            TypeAlias::Unresolved(def) => {
                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::TypeAlias(TypeAlias::Resolving(def.clone())),
                );

                let type_id = self.lower_type(&def.type_ann)?;

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::TypeAlias(TypeAlias::Resolved(type_id)),
                );

                Ok(())
            }
            TypeAlias::Resolving(def) => Err(TypeError::new(
                format!("circular dependency for type alias `{name}`"),
                def.span,
            )),
            TypeAlias::Resolved(_) => Ok(()),
        }
    }

    fn resolve_extern_static_decl(
        &mut self,
        name: &str,
        obj: ScopeObject,
    ) -> Result<(), TypeError> {
        let ScopeObject::ExternStatic(decl) = obj else {
            unreachable!()
        };

        match decl {
            ExternStatic::Unresolved(item) => {
                let ast::ItemKind::Extern(ast::ExternItem::Static(decl)) = &item.kind else {
                    unreachable!()
                };
                let mut symbol_override = None;
                for attr in &item.attrs {
                    match &attr.kind {
                        ast::AttrKind::Symbol(s) => {
                            if symbol_override.replace(s.clone()).is_some() {
                                return Err(TypeError::new("duplicate @symbol".into(), attr.span));
                            }
                        }
                    }
                }
                let symbol = symbol_override
                    .unwrap_or_else(|| self.make_ident(&decl.path).replace("::", "."));

                let type_id = self.lower_type(&decl.type_ann)?;

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::ExternStatic(ExternStatic::Resolved { type_id, symbol }),
                );
                Ok(())
            }
            ExternStatic::Resolved { .. } => Ok(()),
        }
    }

    fn resolve_extern_function_decl(
        &mut self,
        name: &str,
        obj: ScopeObject,
    ) -> Result<(), TypeError> {
        let ScopeObject::ExternFunction(decl) = obj else {
            unreachable!()
        };

        match decl {
            ExternFunction::Unresolved(item) => {
                let ast::ItemKind::Extern(ast::ExternItem::Function(decl)) = &item.kind else {
                    unreachable!()
                };

                let mut symbol_override = None;
                for attr in &item.attrs {
                    match &attr.kind {
                        ast::AttrKind::Symbol(s) => {
                            if symbol_override.replace(s.clone()).is_some() {
                                return Err(TypeError::new("duplicate @symbol".into(), attr.span));
                            }
                        }
                    }
                }
                let symbol = symbol_override
                    .unwrap_or_else(|| self.make_ident(&decl.path).replace("::", "."));

                let params = decl
                    .params
                    .iter()
                    .map(|p| self.lower_type(&p.type_ann))
                    .collect::<Result<Vec<_>, _>>()?;

                let return_ty = self.lower_type(&decl.return_type_ann)?;

                self.scopes.first_mut().insert(
                    name.to_string(),
                    ScopeObject::ExternFunction(ExternFunction::Resolved {
                        params,
                        ret: return_ty,
                        is_variadic: decl.is_variadic,
                        symbol,
                    }),
                );
                Ok(())
            }
            ExternFunction::Resolved { .. } => Ok(()),
        }
    }

    fn lower_items(&mut self, prog: &ast::Program) -> Result<hir::Program, TypeError> {
        let mut uses = Vec::new();
        let mut typed_items = Vec::new();

        for item in &prog.items {
            let typed_item_kind = match &item.kind {
                ast::ItemKind::Use(use_item) => {
                    uses.push(use_item.path.to_string());
                    continue;
                }
                ast::ItemKind::Const(def) => {
                    let name = def.path.to_string();
                    self.resolve_declaration(&name)?;
                    let value = match self.scopes.lookup(&name) {
                        Some(ScopeObject::Const(Const::Resolved(value))) => value.clone(),
                        _ => unreachable!(),
                    };

                    hir::ItemKind::Const(hir::ConstDef {
                        ident: self.make_ident(&def.path),
                        init: value,
                        span: def.span,
                    })
                }
                ast::ItemKind::Static(def) => {
                    let name = def.path.to_string();
                    self.resolve_declaration(&name)?;
                    let (value, symbol) = match self.scopes.lookup(&name) {
                        Some(ScopeObject::Static(Static::Resolved { value, symbol })) => {
                            (value.clone(), symbol.clone())
                        }
                        _ => unreachable!(),
                    };

                    hir::ItemKind::Static(hir::StaticDef {
                        ident: self.make_ident(&def.path),
                        symbol,
                        init: value,
                        span: def.span,
                    })
                }
                ast::ItemKind::Function(_) => hir::ItemKind::Function(self.lower_item_fn(item)?),
                ast::ItemKind::Extern(extern_item) => {
                    let hir_extern = match extern_item {
                        ast::ExternItem::Static(_) => {
                            hir::ExternItem::Static(self.lower_item_extern_static(item)?)
                        }
                        ast::ExternItem::Function(_) => {
                            hir::ExternItem::Function(self.lower_item_extern_fn(item)?)
                        }
                    };
                    hir::ItemKind::Extern(hir_extern)
                }
                ast::ItemKind::Struct(def) => {
                    let name = def.path.to_string();
                    self.resolve_declaration(&name)?;
                    let type_id = match self.scopes.lookup(&name) {
                        Some(ScopeObject::Struct(Struct::Resolved(type_id))) => *type_id,
                        _ => unreachable!(),
                    };

                    hir::ItemKind::TypeDef(hir::TypeDef {
                        ident: self.make_ident(&def.path),
                        type_id,
                        span: def.span,
                    })
                }
                ast::ItemKind::TypeAlias(def) => {
                    let name = def.path.to_string();
                    self.resolve_declaration(&name)?;
                    let type_id = match self.scopes.lookup(&name) {
                        Some(ScopeObject::TypeAlias(TypeAlias::Resolved(type_id))) => *type_id,
                        _ => unreachable!(),
                    };

                    hir::ItemKind::TypeAlias(hir::TypeAlias {
                        ident: self.make_ident(&def.path),
                        type_id,
                        span: def.span,
                    })
                }
            };

            typed_items.push(hir::Item {
                vis: hir::Visibility::from(&item.vis),
                kind: typed_item_kind,
                span: item.span,
            });
        }

        Ok(hir::Program {
            uses,
            items: typed_items,
        })
    }

    fn lower_item_fn(&mut self, item: &ast::Item) -> Result<hir::FunctionDef, TypeError> {
        let ast::ItemKind::Function(func) = &item.kind else {
            unreachable!()
        };

        self.resolve_declaration(&func.path.to_string())?;

        let ident = self.make_ident(&func.path);
        let symbol = match self.scopes.lookup(&func.path.to_string()) {
            Some(ScopeObject::Function(Function::Resolved { symbol, .. })) => symbol.clone(),
            _ => unreachable!(),
        };
        let return_type_id = self.lower_type(&func.return_type_ann)?;
        self.scopes.push(Scope {
            kind: ScopeKind::Function { return_type_id },
            objects: HashMap::new(),
        });

        let mut hir_params = Vec::new();
        for param in &func.params {
            let param_type_id = self.lower_type(&param.type_ann)?;
            let obj = ScopeObject::Var(param_type_id);
            if self
                .scopes
                .last_mut()
                .insert(param.name.clone(), obj)
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

        let typed_body = match &func.body {
            Some(block) => {
                let block = self.check_block(block, return_type_id)?;
                Some(block)
            }
            None => None,
        };

        self.scopes.pop();

        Ok(hir::FunctionDef {
            ident,
            symbol,
            params: hir_params,
            return_type_id,
            body: typed_body,
            span: func.span,
        })
    }

    fn lower_item_extern_fn(
        &mut self,
        item: &ast::Item,
    ) -> Result<hir::ExternFunctionDecl, TypeError> {
        let ast::ItemKind::Extern(ast::ExternItem::Function(decl)) = &item.kind else {
            unreachable!()
        };

        self.resolve_declaration(&decl.path.to_string())?;

        let (param_types, return_type_id, is_variadic, symbol) =
            match self.scopes.lookup(&decl.path.to_string()) {
                Some(ScopeObject::ExternFunction(ExternFunction::Resolved {
                    params,
                    ret,
                    is_variadic,
                    symbol,
                })) => (params.clone(), *ret, *is_variadic, symbol.clone()),
                _ => unreachable!(),
            };
        let params = decl
            .params
            .iter()
            .zip(param_types.iter())
            .map(|(p, &type_id)| hir::Param {
                name: p.name.clone(),
                type_id,
                span: p.span,
            })
            .collect();

        Ok(hir::ExternFunctionDecl {
            ident: self.make_ident(&decl.path),
            symbol,
            params,
            return_type_id,
            is_variadic,
            span: decl.span,
        })
    }

    fn lower_item_extern_static(
        &mut self,
        item: &ast::Item,
    ) -> Result<hir::ExternStaticDecl, TypeError> {
        let ast::ItemKind::Extern(ast::ExternItem::Static(decl)) = &item.kind else {
            unreachable!()
        };

        self.resolve_declaration(&decl.path.to_string())?;

        let (type_id, symbol) = match self.scopes.lookup(&decl.path.to_string()) {
            Some(ScopeObject::ExternStatic(ExternStatic::Resolved { type_id, symbol })) => {
                (*type_id, symbol.clone())
            }
            _ => unreachable!(),
        };

        Ok(hir::ExternStaticDecl {
            ident: self.make_ident(&decl.path),
            symbol,
            type_id,
            span: decl.span,
        })
    }

    fn infer_block(&mut self, block: &ast::Block) -> Result<hir::Block, TypeError> {
        self.scopes.push(Scope {
            kind: ScopeKind::Block,
            objects: HashMap::new(),
        });

        let mut typed_stmts = Vec::new();
        let mut has_never = false;
        let last_idx = block.stmts.len().checked_sub(1);

        for (idx, stmt) in block.stmts.iter().enumerate() {
            if has_never {
                return Err(TypeError::new(
                    "unreachable statement after diverging expression".to_string(),
                    stmt.span,
                ));
            }

            let is_last = Some(idx) == last_idx;

            let typed_stmt = self.infer_statement(stmt)?;

            match &typed_stmt.kind {
                hir::StmtKind::Expr(expr)
                    if !is_last
                        && expr.type_id != TypeId::Unit
                        && expr.type_id != TypeId::Never =>
                {
                    return Err(TypeError::new(
                        format!(
                            "expected `;` after expression: expected `()`, found `{}`",
                            self.types.type_name(expr.type_id)
                        ),
                        expr.span,
                    ));
                }
                hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr)
                    if expr.type_id == TypeId::Never =>
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
            }) if expr.type_id == TypeId::Never => TypeId::Never,
            _ => TypeId::Unit,
        };

        self.scopes.pop();

        Ok(hir::Block {
            stmts: typed_stmts,
            type_id: block_type_id,
            span: block.span,
        })
    }

    fn infer_statement(&mut self, stmt: &ast::Stmt) -> Result<hir::Stmt, TypeError> {
        let kind = match &stmt.kind {
            ast::StmtKind::Let(let_stmt) => {
                let (type_id, typed_init) = if let Some(type_ann) = &let_stmt.type_ann {
                    let declared_type_id = self.lower_type(type_ann)?;
                    let typed_init = self.check_expression(&let_stmt.init, declared_type_id)?;
                    (declared_type_id, typed_init)
                } else {
                    let typed_init = self.infer_expression(&let_stmt.init)?;
                    (typed_init.type_id, typed_init)
                };

                let obj = ScopeObject::Var(type_id);
                if self
                    .scopes
                    .last_mut()
                    .insert(let_stmt.name.clone(), obj)
                    .is_some()
                {
                    return Err(TypeError::new(
                        format!("variable `{}` already defined", let_stmt.name),
                        let_stmt.span,
                    ));
                }

                hir::StmtKind::Let(hir::Let {
                    name: let_stmt.name.clone(),
                    type_id,
                    init: typed_init,
                    span: let_stmt.span,
                })
            }
            ast::StmtKind::Semi(expr) => {
                let typed_expr = self.infer_expression(expr)?;
                hir::StmtKind::Semi(typed_expr)
            }
            ast::StmtKind::Expr(expr) => {
                let typed_expr = self.infer_expression(expr)?;
                hir::StmtKind::Expr(typed_expr)
            }
        };

        Ok(hir::Stmt {
            kind,
            span: stmt.span,
        })
    }

    fn infer_expression(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        match &expr.kind {
            ast::ExprKind::Literal(..) => self.infer_expr_literal(expr),
            ast::ExprKind::Struct(..) => self.infer_expr_struct(expr),
            ast::ExprKind::Path(..) => self.infer_expr_path(expr),
            ast::ExprKind::Array(..) => self.infer_expr_array(expr),
            ast::ExprKind::Repeat(..) => self.infer_expr_repeat(expr),
            ast::ExprKind::Field(..) => self.infer_expr_field(expr),
            ast::ExprKind::Index(..) => self.infer_expr_index(expr),
            ast::ExprKind::Call(..) => self.infer_expr_call(expr),
            ast::ExprKind::Unary(..) => self.infer_expr_unary(expr),
            ast::ExprKind::Binary(..) => self.infer_expr_binary(expr),
            ast::ExprKind::Assign(..) => self.infer_expr_assign(expr),
            ast::ExprKind::Cast(..) => self.infer_expr_cast(expr),
            ast::ExprKind::Return(..) => self.infer_expr_return(expr),
            ast::ExprKind::Block(..) => self.infer_expr_block(expr),
            ast::ExprKind::If(..) => self.infer_expr_if(expr),
            ast::ExprKind::While(..) => self.infer_expr_while(expr),
            ast::ExprKind::Loop(..) => self.infer_expr_loop(expr),
            ast::ExprKind::Break(..) => self.infer_expr_break(expr),
            ast::ExprKind::Continue => self.infer_expr_continue(expr),
        }
    }

    fn infer_expr_literal(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        let (ty, kind) = match lit {
            ast::Literal::Integer(n, suffix) => {
                let (min, max, type_id) = match suffix.as_deref() {
                    Some("u8") => (0i64, i64::from(u8::MAX), TypeId::U8),
                    Some("u16") => (0i64, i64::from(u16::MAX), TypeId::U16),
                    Some("u32") => (0i64, i64::from(u32::MAX), TypeId::U32),
                    Some("u64") => (0i64, u64::MAX.cast_signed(), TypeId::U64),
                    Some("i8") => (0i64, i64::from(i8::MAX), TypeId::I8),
                    Some("i16") => (0i64, i64::from(i16::MAX), TypeId::I16),
                    Some("i32") => (0i64, i64::from(i32::MAX), TypeId::I32),
                    Some("i64") | None => (0i64, i64::MAX, TypeId::I64),
                    Some(unknown) => {
                        return Err(TypeError::new(
                            format!("Unknown integer suffix `{unknown}`"),
                            expr.span,
                        ));
                    }
                };
                if *n < min || *n > max {
                    return Err(TypeError::new(
                        format!(
                            "integer literal `{n}` is out of range for type `{}` ({min}..={max})",
                            self.types.type_name(type_id)
                        ),
                        expr.span,
                    ));
                }

                (type_id, hir::ExprKind::Literal(hir::Literal::Integer(*n)))
            }
            ast::Literal::Float(n, suffix) => {
                let type_id = match suffix.as_deref() {
                    Some("f32") => TypeId::F32,
                    Some("f64") | None => TypeId::F64,
                    Some(unknown) => {
                        return Err(TypeError::new(
                            format!("Unknown float suffix `{unknown}`"),
                            expr.span,
                        ));
                    }
                };
                (type_id, hir::ExprKind::Literal(hir::Literal::Float(*n)))
            }
            ast::Literal::String(s) => (
                self.types.mk_slice(TypeId::U8),
                hir::ExprKind::Literal(hir::Literal::String(s.clone())),
            ),
            ast::Literal::CString(s) => (
                self.types.mk_pointer(TypeId::I8), // char* is signed in GCC and chibicc
                hir::ExprKind::Literal(hir::Literal::CString(s.clone())),
            ),
            ast::Literal::Bool(b) => (TypeId::Bool, hir::ExprKind::Literal(hir::Literal::Bool(*b))),
            ast::Literal::Null => (TypeId::Null, hir::ExprKind::Literal(hir::Literal::Null)),
        };

        Ok(hir::Expr {
            kind,
            type_id: ty,
            span: expr.span,
        })
    }

    fn infer_expr_struct(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Struct(struct_expr) = &expr.kind else {
            unreachable!()
        };

        let path = &struct_expr.path;
        let struct_type_id = match self.scopes.lookup(&path.to_string()) {
            Some(
                ScopeObject::Struct(Struct::Resolved(type_id))
                | ScopeObject::TypeAlias(TypeAlias::Resolved(type_id)),
            ) => *type_id,
            _ => {
                return Err(TypeError::new(
                    format!("undefined struct `{path}`"),
                    expr.span,
                ));
            }
        };

        let expected_fields = match &self.types.get(struct_type_id).kind {
            TypeKind::Struct(_, fields) => fields.clone(),
            _ => {
                return Err(TypeError::new(
                    format!("`{path}` is not a struct"),
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
                        "missing field `{}` in struct literal for `{path}`",
                        field.name
                    ),
                    expr.span,
                )
            })?;

            let typed_value = self.check_expression(&field_init.value, field.type_id)?;

            typed_fields.push(hir::FieldInit {
                name: field.name.clone(),
                value: Box::new(typed_value),
                span: field_init.span,
            });
        }

        if let Some((field_name, field_init)) = provided_fields.into_iter().next() {
            return Err(TypeError::new(
                format!("struct `{path}` has no field `{field_name}`"),
                field_init.span,
            ));
        }

        let ident = self.make_ident(path);

        Ok(hir::Expr {
            kind: hir::ExprKind::Struct(hir::StructExpr {
                ident,
                fields: typed_fields,
                span: struct_expr.span,
            }),
            type_id: struct_type_id,
            span: expr.span,
        })
    }

    fn infer_expr_path(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Path(path) = &expr.kind else {
            unreachable!()
        };

        let name = path.to_string();
        self.resolve_declaration(&name)?;

        match self.scopes.lookup(&name) {
            Some(ScopeObject::Var(type_id)) => Ok(hir::Expr {
                kind: hir::ExprKind::Place(hir::Place::Local(name)),
                type_id: *type_id,
                span: expr.span,
            }),
            Some(ScopeObject::Static(Static::Resolved { value, symbol })) => Ok(hir::Expr {
                kind: hir::ExprKind::Place(hir::Place::Global(symbol.clone())),
                type_id: value.type_id,
                span: expr.span,
            }),
            Some(ScopeObject::ExternStatic(ExternStatic::Resolved { type_id, symbol })) => {
                Ok(hir::Expr {
                    kind: hir::ExprKind::Place(hir::Place::Global(symbol.clone())),
                    type_id: *type_id,
                    span: expr.span,
                })
            }
            Some(ScopeObject::Const(Const::Resolved(val))) => Ok(hir::Expr {
                kind: hir::ExprKind::Const(val.clone()),
                type_id: val.type_id,
                span: expr.span,
            }),
            Some(ScopeObject::Function(Function::Resolved {
                params,
                ret,
                symbol,
            })) => {
                let fn_type_id = self.types.mk_fn(params.clone(), *ret, false);
                Ok(hir::Expr {
                    kind: hir::ExprKind::Place(hir::Place::Global(symbol.clone())),
                    type_id: fn_type_id,
                    span: expr.span,
                })
            }
            Some(ScopeObject::ExternFunction(ExternFunction::Resolved {
                params,
                ret,
                is_variadic,
                symbol,
            })) => {
                let fn_type_id = self.types.mk_fn(params.clone(), *ret, *is_variadic);
                Ok(hir::Expr {
                    kind: hir::ExprKind::Place(hir::Place::Global(symbol.clone())),
                    type_id: fn_type_id,
                    span: expr.span,
                })
            }
            _ => Err(TypeError::new(
                format!("undefined variable `{path}`"),
                expr.span,
            )),
        }
    }

    fn infer_expr_array(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
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
        let first_elem = self.infer_expression(&elems[0])?;
        let elem_type_id = first_elem.type_id;
        typed_elems.push(first_elem);

        for elem in &elems[1..] {
            let typed_elem = self.check_expression(elem, elem_type_id)?;
            typed_elems.push(typed_elem);
        }

        let type_id = self.types.mk_array(elem_type_id, elems.len());
        Ok(hir::Expr {
            kind: hir::ExprKind::Array(typed_elems),
            type_id,
            span: expr.span,
        })
    }

    fn infer_expr_repeat(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let typed_elem = self.infer_expression(elem)?;
        let typed_count = self.check_expression(count, TypeId::I64)?;

        let evaluated_count = self.eval_const_expr(&typed_count)?;

        if let hir::ConstValKind::Integer(n) = evaluated_count.kind {
            Ok(hir::Expr {
                kind: hir::ExprKind::Repeat(Box::new(typed_elem.clone()), n as usize),
                type_id: self.types.mk_array(typed_elem.type_id, n as usize),
                span: expr.span,
            })
        } else {
            Err(TypeError::new(
                "repeat count must be a constant integer".to_string(),
                count.span,
            ))
        }
    }

    fn infer_expr_field(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Field(base, field_name) = &expr.kind else {
            unreachable!()
        };

        let typed_base = self.infer_expression(base)?;

        let (final_base, struct_type_id) = match self.types.get(typed_base.type_id).kind {
            TypeKind::Pointer(inner_type_id) => {
                if !matches!(self.types.get(inner_type_id).kind, TypeKind::Struct(_, _)) {
                    return Err(TypeError::new(
                        format!(
                            "cannot access field on pointer to non-struct type `{}`",
                            self.types.type_name(inner_type_id)
                        ),
                        base.span,
                    ));
                }

                let deref_expr = hir::Expr {
                    kind: hir::ExprKind::Unary(hir::UnaryOp::Deref, Box::new(typed_base)),
                    type_id: inner_type_id,
                    span: base.span,
                };

                (deref_expr, inner_type_id)
            }
            TypeKind::Struct(_, _) => {
                let type_id = typed_base.type_id;
                (typed_base, type_id)
            }
            _ => {
                return Err(TypeError::new(
                    format!(
                        "cannot access field on non-struct type `{}`",
                        self.types.type_name(typed_base.type_id)
                    ),
                    base.span,
                ));
            }
        };

        let TypeKind::Struct(_, fields) = &self.types.get(struct_type_id).kind else {
            unreachable!()
        };

        let field_info = fields
            .iter()
            .find(|field| &field.name == field_name)
            .ok_or_else(|| {
                TypeError::new(format!("no field `{field_name}` on struct"), expr.span)
            })?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Field(Box::new(final_base), field_name.clone()),
            type_id: field_info.type_id,
            span: expr.span,
        })
    }

    fn infer_expr_index(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Index(array, index) = &expr.kind else {
            unreachable!()
        };

        let typed_array = self.infer_expression(array)?;
        let typed_index = self.check_expression(index, TypeId::I64)?;

        let (final_array, elem_type_id) = match self.types.get(typed_array.type_id).kind {
            TypeKind::Pointer(inner_type_id) => match self.types.get(inner_type_id).kind {
                TypeKind::Array(elem, _) | TypeKind::Slice(elem) => {
                    let deref_expr = hir::Expr {
                        kind: hir::ExprKind::Unary(hir::UnaryOp::Deref, Box::new(typed_array)),
                        type_id: inner_type_id,
                        span: array.span,
                    };
                    (deref_expr, elem)
                }
                _ => {
                    return Err(TypeError::new(
                        format!(
                            "cannot index into pointer to non-indexable type `{}`",
                            self.types.type_name(inner_type_id)
                        ),
                        array.span,
                    ));
                }
            },
            TypeKind::Array(elem_type_id, _) | TypeKind::Slice(elem_type_id) => {
                (typed_array, elem_type_id)
            }
            _ => {
                return Err(TypeError::new(
                    format!(
                        "cannot index into type `{}`",
                        self.types.type_name(typed_array.type_id)
                    ),
                    array.span,
                ));
            }
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Index(Box::new(final_array), Box::new(typed_index)),
            type_id: elem_type_id,
            span: expr.span,
        })
    }

    fn infer_expr_call(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Call(call) = &expr.kind else {
            unreachable!()
        };

        let typed_callee = self.infer_expression(&call.callee)?;

        let TypeKind::Fn(params, return_ty, is_variadic) =
            self.types.get(typed_callee.type_id).kind.clone()
        else {
            return Err(TypeError::new(
                "expression is not callable".to_string(),
                call.callee.span,
            ));
        };

        let min_args = params.len();
        if is_variadic {
            if call.args.len() < min_args {
                return Err(TypeError::new(
                    format!(
                        "function expects at least {min_args} arguments, got {}",
                        call.args.len()
                    ),
                    call.span,
                ));
            }
        } else if call.args.len() != min_args {
            return Err(TypeError::new(
                format!(
                    "function expects {min_args} arguments, got {}",
                    call.args.len()
                ),
                call.span,
            ));
        }

        let mut typed_args = Vec::new();
        for (arg, &param_type_id) in call.args.iter().zip(&params) {
            let typed_arg = self.check_expression(arg, param_type_id)?;
            typed_args.push(typed_arg);
        }
        for arg in call.args.iter().skip(min_args) {
            typed_args.push(self.infer_expression(arg)?);
        }

        let variadic_start = is_variadic.then_some(min_args as u64);

        Ok(hir::Expr {
            kind: hir::ExprKind::Call(hir::Call {
                callee: Box::new(typed_callee),
                args: typed_args,
                variadic_start,
                span: call.span,
            }),
            type_id: return_ty,
            span: expr.span,
        })
    }

    fn infer_expr_unary(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Unary(op, operand) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::UnaryOp::from(op);
        let typed_operand = self.infer_expression(operand)?;

        let ty = match typed_op {
            hir::UnaryOp::Neg => {
                let kind = &self.types.get(typed_operand.type_id).kind;
                if !kind.is_signed() && !kind.is_float() {
                    return Err(TypeError::new(
                        format!(
                            "cannot apply `-` to type `{}`",
                            self.types.type_name(typed_operand.type_id)
                        ),
                        operand.span,
                    ));
                }
                typed_operand.type_id
            }
            hir::UnaryOp::Not => {
                let kind = &self.types.get(typed_operand.type_id).kind;
                if !matches!(kind, TypeKind::Bool) && !kind.is_integer() {
                    return Err(TypeError::new(
                        format!(
                            "cannot apply `!` to type `{}`",
                            self.types.type_name(typed_operand.type_id)
                        ),
                        operand.span,
                    ));
                }
                typed_operand.type_id
            }
            hir::UnaryOp::Ref => {
                match &typed_operand.kind {
                    hir::ExprKind::Literal(lit) => {
                        return Err(TypeError::new(
                            format!("cannot take address of constant `{lit:?}`"),
                            operand.span,
                        ));
                    }
                    hir::ExprKind::Const(val) => {
                        return Err(TypeError::new(
                            format!("cannot take address of constant `{val:?}`"),
                            operand.span,
                        ));
                    }
                    _ => {}
                }

                self.types.mk_pointer(typed_operand.type_id)
            }
            hir::UnaryOp::Deref => match self.types.get(typed_operand.type_id).kind {
                TypeKind::Pointer(elem) => elem,
                _ => {
                    return Err(TypeError::new(
                        format!(
                            "cannot dereference non-pointer type `{}`",
                            self.types.type_name(typed_operand.type_id)
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

    fn infer_expr_binary(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Binary(op, left, right) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::BinaryOp::from(op);
        let typed_left = self.infer_expression(left)?;
        let typed_right = self.infer_expression(right)?;

        let ty = match typed_op {
            hir::BinaryOp::Add | hir::BinaryOp::Sub | hir::BinaryOp::Mul | hir::BinaryOp::Div => {
                let lk = &self.types.get(typed_left.type_id).kind;
                if !lk.is_numeric() || typed_left.type_id != typed_right.type_id {
                    return Err(TypeError::new(
                        format!(
                            "arithmetic operator requires numeric operands, found `{}` and `{}`",
                            self.types.type_name(typed_left.type_id),
                            self.types.type_name(typed_right.type_id)
                        ),
                        expr.span,
                    ));
                }
                typed_left.type_id
            }
            hir::BinaryOp::Rem | hir::BinaryOp::BitAnd | hir::BinaryOp::BitOr => {
                let lk = &self.types.get(typed_left.type_id).kind;
                if !lk.is_integer() || typed_left.type_id != typed_right.type_id {
                    return Err(TypeError::new(
                        format!(
                            "arithmetic operator requires integer operands, found `{}` and `{}`",
                            self.types.type_name(typed_left.type_id),
                            self.types.type_name(typed_right.type_id)
                        ),
                        expr.span,
                    ));
                }
                typed_left.type_id
            }
            hir::BinaryOp::Lt | hir::BinaryOp::Le | hir::BinaryOp::Gt | hir::BinaryOp::Ge => {
                let lk = &self.types.get(typed_left.type_id).kind;
                if !lk.is_numeric() || typed_left.type_id != typed_right.type_id {
                    return Err(TypeError::new(
                        format!(
                            "comparison operator requires numeric operands, found `{}` and `{}`",
                            self.types.type_name(typed_left.type_id),
                            self.types.type_name(typed_right.type_id)
                        ),
                        expr.span,
                    ));
                }
                TypeId::Bool
            }
            hir::BinaryOp::Eq | hir::BinaryOp::Ne => {
                let is_null = |id: TypeId| id == TypeId::Null;
                let is_ptr = |id: TypeId| matches!(self.types.get(id).kind, TypeKind::Pointer(_));

                // T == T
                // *T == null
                // null = *T
                if typed_left.type_id == typed_right.type_id
                    || (is_ptr(typed_left.type_id) && is_null(typed_right.type_id))
                    || (is_null(typed_left.type_id) && is_ptr(typed_right.type_id))
                {
                    TypeId::Bool
                } else {
                    return Err(TypeError::new(
                        format!(
                            "equality operator requires same types, found `{}` and `{}`",
                            self.types.type_name(typed_left.type_id),
                            self.types.type_name(typed_right.type_id)
                        ),
                        expr.span,
                    ));
                }
            }
            hir::BinaryOp::And | hir::BinaryOp::Or => {
                if typed_left.type_id != TypeId::Bool || typed_right.type_id != TypeId::Bool {
                    return Err(TypeError::new(
                        format!(
                            "logical operator requires `bool` operands, found `{}` and `{}`",
                            self.types.type_name(typed_left.type_id),
                            self.types.type_name(typed_right.type_id)
                        ),
                        expr.span,
                    ));
                }
                TypeId::Bool
            }
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::Binary(typed_op, Box::new(typed_left), Box::new(typed_right)),
            type_id: ty,
            span: expr.span,
        })
    }

    fn infer_expr_assign(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let typed_lhs = self.infer_expression(lhs)?;
        let typed_rhs = self.check_expression(rhs, typed_lhs.type_id)?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Assign(Box::new(typed_lhs), Box::new(typed_rhs)),
            type_id: TypeId::Unit,
            span: expr.span,
        })
    }

    fn infer_expr_cast(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Cast(lhs, ty) = &expr.kind else {
            unreachable!()
        };

        let typed_lhs = self.infer_expression(lhs)?;
        let target_type_id = self.lower_type(ty)?;

        let from_kind = &self.types.get(typed_lhs.type_id).kind;
        let to_kind = &self.types.get(target_type_id).kind;

        #[allow(clippy::match_like_matches_macro)]
        let valid = match (from_kind, to_kind) {
            _ if from_kind.is_numeric() && to_kind.is_numeric() => true,
            _ => false,
        };

        if !valid {
            return Err(TypeError::new(
                format!(
                    "cannot cast `{}` to `{}`",
                    self.types.type_name(typed_lhs.type_id),
                    self.types.type_name(target_type_id),
                ),
                expr.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Cast(Box::new(typed_lhs), target_type_id),
            type_id: target_type_id,
            span: expr.span,
        })
    }

    fn infer_expr_return(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Return(val) = &expr.kind else {
            unreachable!()
        };

        let return_type_id = *self
            .scopes
            .find_map(|s| match &s.kind {
                ScopeKind::Function { return_type_id, .. } => Some(return_type_id),
                _ => None,
            })
            .ok_or_else(|| TypeError::new("return outside of function".to_string(), expr.span))?;

        let kind = if let Some(v) = val {
            let typed_val = self.check_expression(v, return_type_id)?;
            hir::ExprKind::Return(Some(Box::new(typed_val)))
        } else {
            if return_type_id != TypeId::Unit {
                return Err(TypeError::new(
                    format!(
                        "expected return value of type `{}`",
                        self.types.type_name(return_type_id)
                    ),
                    expr.span,
                ));
            }
            hir::ExprKind::Return(None)
        };

        Ok(hir::Expr {
            kind,
            type_id: TypeId::Never,
            span: expr.span,
        })
    }

    fn infer_expr_block(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Block(block) = &expr.kind else {
            unreachable!()
        };

        let typed_block = self.infer_block(block)?;
        let type_id = typed_block.type_id;

        Ok(hir::Expr {
            kind: hir::ExprKind::Block(typed_block),
            type_id,
            span: expr.span,
        })
    }

    fn infer_expr_if(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&if_expr.cond, TypeId::Bool)?;

        let typed_then = self.infer_block(&if_expr.then_body)?;

        let (ty, typed_else) = if let Some(else_expr) = &if_expr.else_body {
            let typed_else = self.infer_expression(else_expr)?;

            let result_ty = match (typed_then.type_id, typed_else.type_id) {
                (TypeId::Never, TypeId::Never) => TypeId::Never,
                (_, TypeId::Never) => typed_then.type_id,
                (TypeId::Never, _) => typed_else.type_id,
                (then_ty, else_ty) if then_ty == else_ty => typed_then.type_id,
                (then_ty, else_ty) => {
                    return Err(TypeError::new(
                        format!(
                            "if-else branches have different types: `{}` and `{}`",
                            self.types.type_name(then_ty),
                            self.types.type_name(else_ty),
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
            (TypeId::Unit, None)
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

    fn infer_expr_while(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        self.scopes.push(Scope {
            kind: ScopeKind::Loop {
                break_type: None,
                allows_break_value: false,
            },
            objects: HashMap::new(),
        });

        let typed_cond = self.check_expression(&while_expr.cond, TypeId::Bool)?;

        let typed_body = self.infer_block(&while_expr.body)?;

        self.scopes.pop();

        Ok(hir::Expr {
            kind: hir::ExprKind::While(hir::While {
                cond: Box::new(typed_cond),
                body: Box::new(typed_body),
                span: while_expr.span,
            }),
            type_id: TypeId::Unit,
            span: expr.span,
        })
    }

    fn infer_expr_loop(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Loop(loop_expr) = &expr.kind else {
            unreachable!()
        };

        self.scopes.push(Scope {
            kind: ScopeKind::Loop {
                break_type: None,
                allows_break_value: true,
            },
            objects: HashMap::new(),
        });

        let typed_body = self.infer_block(&loop_expr.body)?;

        let break_type = match &self.scopes.last_mut().kind {
            ScopeKind::Loop { break_type, .. } => *break_type,
            _ => unreachable!(),
        };
        self.scopes.pop();

        Ok(hir::Expr {
            kind: hir::ExprKind::Loop(hir::Loop {
                body: Box::new(typed_body),
                span: loop_expr.span,
            }),
            type_id: break_type.unwrap_or(TypeId::Never),
            span: expr.span,
        })
    }

    fn infer_expr_break(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Break(val_expr) = &expr.kind else {
            unreachable!()
        };

        let allows_break_value = self
            .scopes
            .find_map(|s| match &s.kind {
                ScopeKind::Loop {
                    allows_break_value, ..
                } => Some(allows_break_value),
                _ => None,
            })
            .ok_or_else(|| TypeError::new("break outside of loop".into(), expr.span))?;

        if !allows_break_value && val_expr.is_some() {
            return Err(TypeError::new(
                "break with value is not allowed in while loop".into(),
                expr.span,
            ));
        }

        let (kind, val_type) = match val_expr {
            Some(val_expr) => {
                let typed = self.infer_expression(val_expr)?;
                let ty = typed.type_id;
                (hir::ExprKind::Break(Some(Box::new(typed))), ty)
            }
            None => (hir::ExprKind::Break(None), TypeId::Unit),
        };

        let Some(Scope {
            kind: ScopeKind::Loop { break_type, .. },
            ..
        }) = self
            .scopes
            .find_mut(|s| matches!(s.kind, ScopeKind::Loop { .. }))
        else {
            unreachable!()
        };

        match *break_type {
            None => *break_type = Some(val_type),
            Some(existing) if existing != val_type => {
                return Err(TypeError::new(
                    format!(
                        "break value type mismatch: expected {}, found {}",
                        self.types.type_name(existing),
                        self.types.type_name(val_type),
                    ),
                    expr.span,
                ));
            }
            _ => {}
        }

        Ok(hir::Expr {
            kind,
            type_id: TypeId::Never,
            span: expr.span,
        })
    }

    fn infer_expr_continue(&mut self, expr: &ast::Expr) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Continue = &expr.kind else {
            unreachable!()
        };

        if self
            .scopes
            .find(|s| matches!(s.kind, ScopeKind::Loop { .. }))
            .is_none()
        {
            return Err(TypeError::new("continue outside of loop".into(), expr.span));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Continue,
            type_id: TypeId::Never,
            span: expr.span,
        })
    }

    fn check_expression(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        match &expr.kind {
            ast::ExprKind::Literal(..) => self.check_expr_literal(expr, expected),
            ast::ExprKind::Array(..) => self.check_expr_array(expr, expected),
            ast::ExprKind::Repeat(..) => self.check_expr_repeat(expr, expected),
            ast::ExprKind::Block(..) => self.check_expr_block(expr, expected),
            ast::ExprKind::If(..) => self.check_expr_if(expr, expected),
            ast::ExprKind::Binary(..) => self.check_expr_binary(expr, expected),
            ast::ExprKind::Unary(..) => self.check_expr_unary(expr, expected),
            _ => {
                let inferred = self.infer_expression(expr)?;
                if !self.types.is_assignable(inferred.type_id, expected) {
                    return Err(TypeError::new(
                        format!(
                            "expected `{}`, found `{}`",
                            self.types.type_name(expected),
                            self.types.type_name(inferred.type_id),
                        ),
                        expr.span,
                    ));
                }
                Ok(inferred)
            }
        }
    }

    fn check_expr_literal(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        match lit {
            ast::Literal::Integer(n, None) if self.types.get(expected).kind.is_integer() => {
                let (min, max, name) = match expected {
                    TypeId::U8 => (0i64, i64::from(u8::MAX), "u8"),
                    TypeId::U16 => (0i64, i64::from(u16::MAX), "u16"),
                    TypeId::U32 => (0i64, i64::from(u32::MAX), "u32"),
                    TypeId::U64 => (0i64, u64::MAX.cast_signed(), "u64"),
                    TypeId::I8 => (i64::from(i8::MIN), i64::from(i8::MAX), "i8"),
                    TypeId::I16 => (i64::from(i16::MIN), i64::from(i16::MAX), "i16"),
                    TypeId::I32 => (i64::from(i32::MIN), i64::from(i32::MAX), "i32"),
                    TypeId::I64 => (i64::MIN, i64::MAX, "i64"),
                    _ => unreachable!(),
                };
                if *n < min || *n > max {
                    return Err(TypeError::new(
                        format!(
                            "integer literal `{n}` is out of range for type `{name}` ({min}..={max})"
                        ),
                        expr.span,
                    ));
                }
                Ok(hir::Expr {
                    kind: hir::ExprKind::Literal(hir::Literal::Integer(*n)),
                    type_id: expected,
                    span: expr.span,
                })
            }
            ast::Literal::Float(n, None) if self.types.get(expected).kind.is_float() => {
                Ok(hir::Expr {
                    kind: hir::ExprKind::Literal(hir::Literal::Float(*n)),
                    type_id: expected,
                    span: expr.span,
                })
            }
            _ => {
                let inferred = self.infer_expr_literal(expr)?;
                if !self.types.is_assignable(inferred.type_id, expected) {
                    return Err(TypeError::new(
                        format!(
                            "expected `{}`, found `{}`",
                            self.types.type_name(expected),
                            self.types.type_name(inferred.type_id),
                        ),
                        expr.span,
                    ));
                }
                Ok(inferred)
            }
        }
    }

    fn check_expr_array(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Array(elems) = &expr.kind else {
            unreachable!()
        };

        let TypeKind::Array(elem_type_id, expected_len) = self.types.get(expected).kind else {
            let inferred = self.infer_expr_repeat(expr)?;
            if !self.types.is_assignable(inferred.type_id, expected) {
                return Err(TypeError::new(
                    format!(
                        "expected `{}`, found `{}`",
                        self.types.type_name(expected),
                        self.types.type_name(inferred.type_id),
                    ),
                    expr.span,
                ));
            }
            return Ok(inferred);
        };

        let mut typed_elems = Vec::new();
        for elem in elems {
            let typed_elem = self.check_expression(elem, elem_type_id)?;
            typed_elems.push(typed_elem);
        }

        if elems.len() != expected_len {
            return Err(TypeError::new(
                format!(
                    "array length mismatch: expected {} elements, found {}",
                    expected_len,
                    elems.len(),
                ),
                expr.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Array(typed_elems),
            type_id: expected,
            span: expr.span,
        })
    }

    fn check_expr_repeat(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let TypeKind::Array(elem_type_id, expected_len) = self.types.get(expected).kind else {
            let inferred = self.infer_expr_repeat(expr)?;
            if !self.types.is_assignable(inferred.type_id, expected) {
                return Err(TypeError::new(
                    format!(
                        "expected `{}`, found `{}`",
                        self.types.type_name(expected),
                        self.types.type_name(inferred.type_id),
                    ),
                    expr.span,
                ));
            }
            return Ok(inferred);
        };

        let typed_elem = self.check_expression(elem, elem_type_id)?;
        let typed_count = self.check_expression(count, TypeId::I64)?;
        let evaluated_count = self.eval_const_expr(&typed_count)?;

        let hir::ConstValKind::Integer(n) = evaluated_count.kind else {
            return Err(TypeError::new(
                "repeat count must be a constant integer".to_string(),
                count.span,
            ));
        };

        if n as usize != expected_len {
            return Err(TypeError::new(
                format!("array length mismatch: expected {expected_len} elements, found {n}",),
                count.span,
            ));
        }

        Ok(hir::Expr {
            kind: hir::ExprKind::Repeat(Box::new(typed_elem), n as usize),
            type_id: expected,
            span: expr.span,
        })
    }

    fn check_expr_block(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Block(block) = &expr.kind else {
            unreachable!()
        };

        let typed_block = self.check_block(block, expected)?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Block(typed_block),
            type_id: expected,
            span: expr.span,
        })
    }
    fn check_expr_if(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let typed_cond = self.check_expression(&if_expr.cond, TypeId::Bool)?;

        let typed_then = self.check_block(&if_expr.then_body, expected)?;

        let typed_else = if let Some(else_expr) = &if_expr.else_body {
            Some(Box::new(self.check_expression(else_expr, expected)?))
        } else {
            // An else-less if expression always has type ()
            if expected != TypeId::Unit {
                return Err(TypeError::new(
                    format!(
                        "`if` without `else` has type `()`, but expected `{}`",
                        self.types.type_name(expected),
                    ),
                    expr.span,
                ));
            }
            None
        };

        Ok(hir::Expr {
            kind: hir::ExprKind::If(hir::If {
                cond: Box::new(typed_cond),
                then_body: Box::new(typed_then),
                else_body: typed_else,
                span: if_expr.span,
            }),
            type_id: expected,
            span: expr.span,
        })
    }

    fn check_block(
        &mut self,
        block: &ast::Block,
        expected: TypeId,
    ) -> Result<hir::Block, TypeError> {
        self.scopes.push(Scope {
            kind: ScopeKind::Block,
            objects: HashMap::new(),
        });

        let mut typed_stmts = Vec::new();
        let mut has_never = false;
        let last_idx = block.stmts.len().checked_sub(1);

        for (idx, stmt) in block.stmts.iter().enumerate() {
            if has_never {
                return Err(TypeError::new(
                    "unreachable code after diverging expression".to_string(),
                    stmt.span,
                ));
            }

            let is_last = Some(idx) == last_idx;

            // only the trailing expression is in checking
            // all other statements are synthesized
            let typed_stmt = match (is_last, &stmt.kind) {
                (true, ast::StmtKind::Expr(expr)) => {
                    let typed_expr = self.check_expression(expr, expected)?;
                    hir::Stmt {
                        kind: hir::StmtKind::Expr(typed_expr),
                        span: stmt.span,
                    }
                }
                _ => self.infer_statement(stmt)?,
            };

            match &typed_stmt.kind {
                hir::StmtKind::Expr(expr)
                    if !is_last
                        && expr.type_id != TypeId::Unit
                        && expr.type_id != TypeId::Never =>
                {
                    return Err(TypeError::new(
                        format!(
                            "expected `;` after expression: expected `()`, found `{}`",
                            self.types.type_name(expr.type_id)
                        ),
                        expr.span,
                    ));
                }
                hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr)
                    if expr.type_id == TypeId::Never =>
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
            }) if expr.type_id == TypeId::Never => TypeId::Never,
            _ => TypeId::Unit,
        };

        if !self.types.is_assignable(block_type_id, expected) {
            return Err(TypeError::new(
                format!(
                    "expected `{}`, found `{}`",
                    self.types.type_name(expected),
                    self.types.type_name(block_type_id),
                ),
                block.span,
            ));
        }

        self.scopes.pop();

        Ok(hir::Block {
            stmts: typed_stmts,
            type_id: block_type_id,
            span: block.span,
        })
    }

    fn check_expr_binary(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Binary(op, left, right) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::BinaryOp::from(op);

        let propagate = match typed_op {
            hir::BinaryOp::Add | hir::BinaryOp::Sub | hir::BinaryOp::Mul | hir::BinaryOp::Div => {
                self.types.get(expected).kind.is_numeric()
            }
            hir::BinaryOp::Rem | hir::BinaryOp::BitAnd | hir::BinaryOp::BitOr => {
                self.types.get(expected).kind.is_integer()
            }
            _ => false,
        };

        if !propagate {
            let inferred = self.infer_expr_binary(expr)?;
            if !self.types.is_assignable(inferred.type_id, expected) {
                return Err(TypeError::new(
                    format!(
                        "expected `{}`, found `{}`",
                        self.types.type_name(expected),
                        self.types.type_name(inferred.type_id),
                    ),
                    expr.span,
                ));
            }
            return Ok(inferred);
        }

        let typed_left = self.check_expression(left, expected)?;
        let typed_right = self.check_expression(right, expected)?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Binary(typed_op, Box::new(typed_left), Box::new(typed_right)),
            type_id: expected,
            span: expr.span,
        })
    }

    fn check_expr_unary(
        &mut self,
        expr: &ast::Expr,
        expected: TypeId,
    ) -> Result<hir::Expr, TypeError> {
        let ast::ExprKind::Unary(op, operand) = &expr.kind else {
            unreachable!()
        };

        let typed_op = hir::UnaryOp::from(op);

        let propagate = match typed_op {
            hir::UnaryOp::Neg => {
                let kind = &self.types.get(expected).kind;
                kind.is_signed() || kind.is_float()
            }
            hir::UnaryOp::Not => {
                let kind = &self.types.get(expected).kind;
                matches!(kind, TypeKind::Bool) || kind.is_integer()
            }
            hir::UnaryOp::Ref | hir::UnaryOp::Deref => false,
        };

        if !propagate {
            let inferred = self.infer_expr_unary(expr)?;
            if !self.types.is_assignable(inferred.type_id, expected) {
                return Err(TypeError::new(
                    format!(
                        "expected `{}`, found `{}`",
                        self.types.type_name(expected),
                        self.types.type_name(inferred.type_id),
                    ),
                    expr.span,
                ));
            }
            return Ok(inferred);
        }

        let typed_operand = self.check_expression(operand, expected)?;

        Ok(hir::Expr {
            kind: hir::ExprKind::Unary(typed_op, Box::new(typed_operand)),
            type_id: expected,
            span: expr.span,
        })
    }
}
