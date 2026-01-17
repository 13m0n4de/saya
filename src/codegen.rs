use std::{error::Error, fmt};

use indexmap::IndexMap;

use crate::{
    hir::*,
    span::Span,
    types::{Field, Type, TypeContext, TypeId, TypeKind},
};

#[derive(Debug, Clone)]
pub enum GenValue {
    Const(u64, TypeId),
    Global(String, TypeId),
    Temp(String, TypeId),
}

impl GenValue {
    pub fn ty(&self) -> &TypeId {
        match self {
            GenValue::Const(_, ty) | GenValue::Global(_, ty) | GenValue::Temp(_, ty) => ty,
        }
    }
}

impl From<GenValue> for qbe::Value {
    fn from(value: GenValue) -> Self {
        match value {
            GenValue::Const(val, _) => qbe::Value::Const(val),
            GenValue::Global(name, _) => qbe::Value::Global(name),
            GenValue::Temp(name, _) => qbe::Value::Temporary(name),
        }
    }
}

#[derive(Debug, Clone)]
struct LoopContext {
    continue_label: String,
    break_label: String,
}

#[derive(Debug)]
pub struct CodeGenError {
    pub message: String,
    pub span: Span,
}

impl CodeGenError {
    pub fn new(message: String, span: Span) -> Self {
        Self { message, span }
    }
}

impl fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "codegen error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl Error for CodeGenError {}

pub struct CodeGen<'a> {
    types: &'a mut TypeContext,
    temp_counter: usize,
    label_counter: usize,
    string_counter: usize,
    loops: Vec<LoopContext>,
    data_defs: Vec<qbe::DataDef<'static>>,
    type_defs: IndexMap<TypeId, &'static qbe::TypeDef<'static>>,
}

impl<'a> CodeGen<'a> {
    pub fn new(types: &'a mut TypeContext) -> Self {
        Self {
            types,
            temp_counter: 0,
            label_counter: 0,
            string_counter: 0,
            loops: Vec::new(),
            data_defs: Vec::new(),
            type_defs: IndexMap::new(),
        }
    }

    pub fn generate(&mut self, prog: &Program) -> Result<String, CodeGenError> {
        let mut module = qbe::Module::new();

        for item in &prog.items {
            match &item.kind {
                // Globals
                ItemKind::Static(static_def) => {
                    self.generate_static(static_def, &item.vis);
                }
                // Functions
                ItemKind::Function(func) if func.body.is_some() => {
                    let qbe_func = self.generate_function(func, &item.vis)?;
                    module.add_function(qbe_func);
                }
                _ => {}
            }
        }

        // TypeDefs
        for &type_def in self.type_defs.values() {
            module.add_type(type_def.clone());
        }

        // DataDefs
        for data_def in &self.data_defs {
            module.add_data(data_def.clone());
        }

        Ok(module.to_string())
    }

    fn ident_to_symbol(ident: &str) -> String {
        ident.replace("::", ".")
    }

    fn qbe_type(&mut self, type_id: TypeId) -> qbe::Type<'static> {
        let ty = self.types.get(type_id);
        match &ty.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::Word,
            TypeKind::Bool => qbe::Type::Word,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
            TypeKind::Struct(_) | TypeKind::Array(..) | TypeKind::Slice(_) => {
                let def = self.generate_type_def(type_id);
                qbe::Type::Aggregate(def)
            }
        }
    }

    fn qbe_load_type(&self, type_id: TypeId) -> qbe::Type<'static> {
        let ty = self.types.get(type_id);
        match &ty.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::UnsignedByte,
            TypeKind::Bool => qbe::Type::UnsignedByte,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
            TypeKind::Struct(_) | TypeKind::Array(..) | TypeKind::Slice(_) => qbe::Type::Long,
        }
    }

    fn qbe_store_type(&self, type_id: TypeId) -> qbe::Type<'static> {
        let ty = self.types.get(type_id);
        match &ty.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::Byte,
            TypeKind::Bool => qbe::Type::Byte,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
            TypeKind::Struct(_) | TypeKind::Array(..) | TypeKind::Slice(_) => qbe::Type::Long,
        }
    }

    fn build_struct_def(
        &mut self,
        type_id: TypeId,
        fields: &[Field],
    ) -> &'static qbe::TypeDef<'static> {
        let ty = self.types.get(type_id);

        let qbe_fields: Vec<_> = fields
            .iter()
            .map(|field| {
                let field_ty = self.types.get(field.type_id);
                let qbe_ty = if field_ty.is_aggregate() {
                    let field_def = self
                        .type_defs
                        .get(&field.type_id)
                        .expect("field type should be generated first");
                    qbe::Type::Aggregate(field_def)
                } else {
                    self.qbe_store_type(field.type_id)
                };
                (qbe_ty, 1)
            })
            .collect();

        let name = format!("type.{}", type_id.0);

        let def = Box::new(qbe::TypeDef {
            name,
            align: Some(ty.align),
            items: qbe_fields,
        });

        Box::leak(def)
    }

    fn build_array_def(
        &mut self,
        type_id: TypeId,
        elem_type_id: TypeId,
        len: usize,
    ) -> &'static qbe::TypeDef<'static> {
        let ty = self.types.get(type_id);
        let elem_ty = self.types.get(elem_type_id);

        let qbe_elem_ty = if elem_ty.is_aggregate() {
            let elem_def = self
                .type_defs
                .get(&elem_type_id)
                .expect("element type should be generated first");
            qbe::Type::Aggregate(elem_def)
        } else {
            self.qbe_store_type(elem_type_id)
        };

        let items = vec![(qbe_elem_ty, len)];

        let name = format!("type.{}", type_id.0);

        let def = Box::new(qbe::TypeDef {
            name,
            align: Some(ty.align),
            items,
        });

        Box::leak(def)
    }

    fn build_slice_def(&mut self, type_id: TypeId) -> &'static qbe::TypeDef<'static> {
        let ty = self.types.get(type_id);

        let items = vec![(qbe::Type::Long, 1), (qbe::Type::Long, 1)];

        let name = format!("type.{}", type_id.0);

        let def = Box::new(qbe::TypeDef {
            name,
            align: Some(ty.align),
            items,
        });

        Box::leak(def)
    }

    fn new_temp(&mut self) -> String {
        let name = format!("temp.{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn assign_to_temp(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        type_id: TypeId,
        instr: qbe::Instr<'static>,
    ) -> GenValue {
        let name = self.new_temp();
        let temp = qbe::Value::Temporary(name.clone());
        let qbe_ty = self.qbe_type(type_id);
        qfunc.assign_instr(temp, qbe_ty, instr);
        GenValue::Temp(name, type_id)
    }

    fn new_label(&mut self) -> usize {
        let id = self.label_counter;
        self.label_counter += 1;
        id
    }

    fn alloc_local(&mut self, qfunc: &mut qbe::Function<'static>, ty: &Type) -> qbe::Value {
        let size = ty.size;
        let align = ty.align;
        let name = self.new_temp();
        let addr = qbe::Value::Temporary(name);

        let alloc_instr = if align >= 16 {
            qbe::Instr::Alloc16(size as u128)
        } else if align >= 8 {
            qbe::Instr::Alloc8(size as u64)
        } else {
            qbe::Instr::Alloc4(size as u32)
        };

        qfunc.assign_instr(addr.clone(), qbe::Type::Long, alloc_instr);
        addr
    }

    fn load_field(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        base_addr: qbe::Value,
        offset: u64,
        type_id: TypeId,
    ) -> qbe::Value {
        let addr = if offset == 0 {
            base_addr
        } else {
            let temp = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                temp.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(base_addr, qbe::Value::Const(offset)),
            );
            temp
        };

        if self.types.get(type_id).is_aggregate() {
            return addr;
        }

        let base_ty = self.qbe_type(type_id);
        let load_ty = self.qbe_load_type(type_id);
        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(result.clone(), base_ty, qbe::Instr::Load(load_ty, addr));
        result
    }

    fn store_field(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        base_addr: qbe::Value,
        offset: u64,
        value: qbe::Value,
        type_id: TypeId,
    ) {
        let addr = if offset == 0 {
            base_addr
        } else {
            let temp = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                temp.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(base_addr, qbe::Value::Const(offset)),
            );
            temp
        };

        if self.types.get(type_id).is_aggregate() {
            self.copy_aggregate(qfunc, addr, value, type_id);
        } else {
            let store_ty = self.qbe_store_type(type_id);
            qfunc.add_instr(qbe::Instr::Store(store_ty, addr, value));
        }
    }

    fn copy_aggregate(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        dest: qbe::Value,
        src: qbe::Value,
        type_id: TypeId,
    ) {
        let type_kind = self.types.get(type_id).kind.clone();
        match type_kind {
            TypeKind::Slice(elem) => {
                // Slice: { ptr: l, len: l }
                let ptr_type_id = self.types.mk_pointer(elem);
                let ptr = self.load_field(qfunc, src.clone(), 0, ptr_type_id);
                let len = self.load_field(qfunc, src, 8, TypeId::I64);
                self.store_field(qfunc, dest.clone(), 0, ptr, ptr_type_id);
                self.store_field(qfunc, dest, 8, len, TypeId::I64);
            }
            TypeKind::Array(elem, len) => {
                // Array: copy each element
                let elem_ty = self.types.get(elem);
                let elem_size = elem_ty.size;

                for i in 0..len {
                    let offset = i as u64 * elem_size as u64;
                    let elem_val = self.load_field(qfunc, src.clone(), offset, elem);
                    self.store_field(qfunc, dest.clone(), offset, elem_val, elem);
                }
            }
            TypeKind::Struct(fields) => {
                // Struct: copy each field
                for field in fields {
                    let field_val =
                        self.load_field(qfunc, src.clone(), field.offset as u64, field.type_id);
                    self.store_field(
                        qfunc,
                        dest.clone(),
                        field.offset as u64,
                        field_val,
                        field.type_id,
                    );
                }
            }
            _ => unreachable!(
                "copy_aggregate called on non-aggregate type: {:?}",
                type_kind
            ),
        }
    }

    fn store_value(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        dest_addr: qbe::Value,
        src: GenValue,
        type_id: TypeId,
    ) {
        if self.types.get(type_id).is_aggregate() {
            self.copy_aggregate(qfunc, dest_addr, src.into(), type_id);
        } else {
            let store_type = self.qbe_store_type(type_id);
            qfunc.add_instr(qbe::Instr::Store(store_type, dest_addr, src.into()));
        }
    }

    fn load_value(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        addr: qbe::Value,
        type_id: TypeId,
    ) -> GenValue {
        if self.types.get(type_id).is_aggregate() {
            match addr {
                qbe::Value::Temporary(name) => GenValue::Temp(name, type_id),
                qbe::Value::Global(name) => GenValue::Global(name, type_id),
                qbe::Value::Const(_) => unreachable!("cannot load from a constant address"),
            }
        } else {
            let load_type = self.qbe_load_type(type_id);
            self.assign_to_temp(qfunc, type_id, qbe::Instr::Load(load_type, addr))
        }
    }

    fn push_loop(&mut self, continue_label: String, break_label: String) {
        self.loops.push(LoopContext {
            continue_label,
            break_label,
        });
    }

    fn pop_loop(&mut self) {
        self.loops
            .pop()
            .expect("ICE: cannot pop loop, loops stack is empty");
    }

    fn current_loop(&self) -> Option<&LoopContext> {
        self.loops.last()
    }

    fn address_of(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<qbe::Value, CodeGenError> {
        match &expr.kind {
            ExprKind::Place(place) => match place {
                // x -> %x
                Place::Local(ident) => {
                    let symbol = Self::ident_to_symbol(ident);
                    Ok(qbe::Value::Temporary(symbol))
                }
                // x -> $x
                Place::Global(ident) => {
                    let symbol = Self::ident_to_symbol(ident);
                    Ok(qbe::Value::Global(symbol))
                }
            },
            // *ptr -> value_of(ptr)
            ExprKind::Unary(UnaryOp::Deref, ptr_expr) => {
                Ok(self.generate_expression(qfunc, ptr_expr)?.into())
            }
            // arr[i] or slice[i] -> calculate element address
            ExprKind::Index(base, index) => {
                let index_val = self.generate_expression(qfunc, index)?.into();
                let base_type_kind = self.types.get(base.type_id).kind.clone();

                let (base_ptr, elem_size) = match base_type_kind {
                    TypeKind::Array(elem, _) => {
                        // Array: base is already the array address
                        let arr_ptr = self.generate_expression(qfunc, base)?.into();
                        (arr_ptr, self.types.get(elem).size)
                    }
                    TypeKind::Slice(elem) => {
                        // Slice: need to load ptr field from slice struct
                        let slice_addr = self.generate_expression(qfunc, base)?.into();
                        let ptr_type_id = self.types.mk_pointer(elem);
                        let ptr = self.load_field(qfunc, slice_addr, 0, ptr_type_id);
                        (ptr, self.types.get(elem).size)
                    }
                    _ => {
                        return Err(CodeGenError::new(
                            format!("Cannot index into type {:?}", base.type_id),
                            expr.span,
                        ));
                    }
                };

                // Calculate offset: index * elem_size
                let offset = qbe::Value::Temporary(self.new_temp());
                qfunc.assign_instr(
                    offset.clone(),
                    qbe::Type::Long,
                    qbe::Instr::Mul(index_val, qbe::Value::Const(elem_size as u64)),
                );

                // Calculate element address: base_ptr + offset
                let elem_addr = qbe::Value::Temporary(self.new_temp());
                qfunc.assign_instr(
                    elem_addr.clone(),
                    qbe::Type::Long,
                    qbe::Instr::Add(base_ptr, offset),
                );

                Ok(elem_addr)
            }
            // base.field -> calculate field address
            ExprKind::Field(base, field_name) => {
                let base_addr = self.address_of(qfunc, base)?;

                let TypeKind::Struct(fields) = &self.types.get(base.type_id).kind else {
                    unreachable!()
                };

                let field_info = fields
                    .iter()
                    .find(|field| &field.name == field_name)
                    .cloned()
                    .expect("field should exist after type checking");

                let field_addr = if field_info.offset == 0 {
                    base_addr
                } else {
                    let temp = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(
                        temp.clone(),
                        qbe::Type::Long,
                        qbe::Instr::Add(base_addr, qbe::Value::Const(field_info.offset as u64)),
                    );
                    temp
                };

                Ok(field_addr)
            }
            _ => Err(CodeGenError::new(
                format!("Cannot take address of this expression: {:?}", expr.kind),
                expr.span,
            )),
        }
    }

    fn literal_to_data_items(&mut self, lit: &Literal) -> Vec<(qbe::Type<'static>, qbe::DataItem)> {
        match lit {
            Literal::Integer(n) => vec![(qbe::Type::Long, qbe::DataItem::Const(n.cast_unsigned()))],
            Literal::Bool(b) => vec![(qbe::Type::Word, qbe::DataItem::Const(u64::from(*b)))],
            Literal::String(s) => {
                let label = self.emit_string_data(s);
                vec![
                    (qbe::Type::Long, qbe::DataItem::Symbol(label, None)), // ptr
                    (qbe::Type::Long, qbe::DataItem::Const(s.len() as u64)), // len
                ]
            }
            Literal::CString(s) => {
                let label = self.emit_cstring_data(s);
                vec![(qbe::Type::Long, qbe::DataItem::Symbol(label, None))]
            }
        }
    }

    fn generate_type_def(&mut self, type_id: TypeId) -> &'static qbe::TypeDef<'static> {
        if let Some(&def) = self.type_defs.get(&type_id) {
            return def;
        }

        let type_kind = self.types.get(type_id).kind.clone();
        let def = match type_kind {
            TypeKind::Struct(fields) => {
                for field in &fields {
                    if self.types.get(field.type_id).is_aggregate() {
                        self.generate_type_def(field.type_id);
                    }
                }

                self.build_struct_def(type_id, &fields)
            }
            TypeKind::Array(elem_type_id, len) => {
                if self.types.get(elem_type_id).is_aggregate() {
                    self.generate_type_def(elem_type_id);
                }

                self.build_array_def(type_id, elem_type_id, len)
            }
            TypeKind::Slice(_) => self.build_slice_def(type_id),
            _ => unreachable!("non-aggregate type: {:?}", type_kind),
        };

        self.type_defs.insert(type_id, def);

        def
    }

    fn generate_static(&mut self, static_def: &StaticDef, vis: &Visibility) {
        let symbol = Self::ident_to_symbol(&static_def.ident);
        let data_items = self.literal_to_data_items(&static_def.init);
        let linkage = match vis {
            Visibility::Public => qbe::Linkage::public(),
            Visibility::Private => qbe::Linkage::private(),
        };

        self.data_defs
            .push(qbe::DataDef::new(linkage, symbol, None, data_items));
    }

    fn generate_function(
        &mut self,
        func: &FunctionDef,
        vis: &Visibility,
    ) -> Result<qbe::Function<'static>, CodeGenError> {
        let Some(block) = &func.body else {
            unreachable!()
        };

        let params: Vec<(qbe::Type<'static>, qbe::Value)> = func
            .params
            .iter()
            .map(|param| {
                let qbe_ty = self.qbe_type(param.type_id);
                let value = qbe::Value::Temporary(format!("{}.param", param.name));
                (qbe_ty, value)
            })
            .collect();

        let qbe_return_type: Option<qbe::Type<'static>> = if func.return_type_id == TypeId::UNIT {
            None
        } else {
            Some(self.qbe_type(func.return_type_id))
        };

        let symbol = Self::ident_to_symbol(&func.ident);
        let linkage = match vis {
            Visibility::Public => qbe::Linkage::public(),
            Visibility::Private => qbe::Linkage::private(),
        };

        let mut qfunc = qbe::Function::new(linkage, symbol, params, qbe_return_type);

        qfunc.add_block("start");

        for param in &func.params {
            let addr = qbe::Value::Temporary(param.name.clone());

            let param_type = self.types.get(param.type_id);
            let param_size = param_type.size;
            let param_align = param_type.align;
            let alloc_instr = if param_align >= 16 {
                qbe::Instr::Alloc16(param_size as u128)
            } else if param_align >= 8 {
                qbe::Instr::Alloc8(param_size as u64)
            } else {
                qbe::Instr::Alloc4(param_size as u32)
            };
            qfunc.assign_instr(addr.clone(), qbe::Type::Long, alloc_instr);

            let param_gen_val = GenValue::Temp(format!("{}.param", param.name), param.type_id);
            self.store_value(&mut qfunc, addr, param_gen_val, param.type_id);
        }

        qfunc.add_block("body");

        let block_value = self.generate_block(&mut qfunc, block)?;

        if func.return_type_id == TypeId::NEVER {
            qfunc.add_instr(qbe::Instr::Hlt);
        } else if block.type_id == TypeId::NEVER {
            if let Some(last_block) = qfunc.blocks.last()
                && !last_block.jumps()
            {
                qfunc.add_instr(qbe::Instr::Hlt);
            }
        } else if block.type_id == TypeId::UNIT {
            qfunc.add_instr(qbe::Instr::Ret(None));
        } else {
            qfunc.add_instr(qbe::Instr::Ret(Some(block_value.into())));
        }

        Ok(qfunc)
    }

    fn generate_block(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        block: &Block,
    ) -> Result<GenValue, CodeGenError> {
        let mut result = GenValue::Const(0, TypeId::UNIT);
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Semi(expr) => {
                    self.generate_expression(qfunc, expr)?;
                }
                StmtKind::Expr(expr) => {
                    result = self.generate_expression(qfunc, expr)?;
                }
                StmtKind::Let(let_stmt) => {
                    self.generate_let(qfunc, let_stmt)?;
                }
            }
        }
        Ok(result)
    }

    fn generate_let(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        let_stmt: &Let,
    ) -> Result<(), CodeGenError> {
        // Allocate space on stack with proper alignment
        let let_stmt_type = self.types.get(let_stmt.type_id);
        let size = let_stmt_type.size;
        let align = let_stmt_type.align;
        let addr = qbe::Value::Temporary(let_stmt.name.clone());

        let alloc_instr = if align >= 16 {
            qbe::Instr::Alloc16(size as u128)
        } else if align >= 8 {
            qbe::Instr::Alloc8(size as u64)
        } else {
            qbe::Instr::Alloc4(size as u32)
        };

        qfunc.assign_instr(addr.clone(), qbe::Type::Long, alloc_instr);

        // Generate initial value
        let init_val = self.generate_expression(qfunc, &let_stmt.init)?;

        // Store value
        self.store_value(qfunc, addr, init_val, let_stmt.type_id);

        Ok(())
    }

    fn generate_expr_literal(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> GenValue {
        let ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        match lit {
            Literal::Integer(n) => GenValue::Const(n.cast_unsigned(), expr.type_id),
            Literal::Bool(b) => GenValue::Const(u64::from(*b), TypeId::BOOL),
            Literal::String(_) => self.generate_string_slice(qfunc, expr),
            Literal::CString(s) => {
                let label = self.emit_cstring_data(s);
                GenValue::Global(label, expr.type_id)
            }
        }
    }

    fn generate_expr_struct(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Struct(struct_expr) = &expr.kind else {
            unreachable!()
        };

        let type_kind = self.types.get(expr.type_id).kind.clone();
        let TypeKind::Struct(fields) = type_kind else {
            unreachable!()
        };

        let ty = self.types.get(expr.type_id).clone();
        let struct_addr = self.alloc_local(qfunc, &ty);

        for field_init in &struct_expr.fields {
            let field_info = fields
                .iter()
                .find(|f| f.name == field_init.name)
                .expect("type checker should ensure all fields exist");

            let field_value = self.generate_expression(qfunc, &field_init.value)?;

            self.store_field(
                qfunc,
                struct_addr.clone(),
                field_info.offset as u64,
                field_value.into(),
                field_info.type_id,
            );
        }

        Ok(match struct_addr {
            qbe::Value::Temporary(name) => GenValue::Temp(name, expr.type_id),
            _ => unreachable!(),
        })
    }

    fn generate_expr_place(&mut self, qfunc: &mut qbe::Function<'static>, expr: &Expr) -> GenValue {
        let ExprKind::Place(place) = &expr.kind else {
            unreachable!()
        };

        match place {
            Place::Local(ident) => {
                let symbol = Self::ident_to_symbol(ident);
                let addr = qbe::Value::Temporary(symbol);
                self.load_value(qfunc, addr, expr.type_id)
            }
            Place::Global(ident) => {
                let symbol = Self::ident_to_symbol(ident);
                let addr = qbe::Value::Global(symbol);
                self.load_value(qfunc, addr, expr.type_id)
            }
        }
    }

    fn generate_expr_unary(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Unary(unop, operand_expr) = &expr.kind else {
            unreachable!()
        };

        let instr = match unop {
            UnaryOp::Neg => {
                let operand = self.generate_expression(qfunc, operand_expr)?.into();
                qbe::Instr::Neg(operand)
            }
            UnaryOp::Not => {
                let operand = self.generate_expression(qfunc, operand_expr)?.into();
                let operand_expr_type = self.types.get(operand_expr.type_id).clone();
                let result_ty = self.qbe_type(expr.type_id);
                match operand_expr_type.kind {
                    TypeKind::Bool => qbe::Instr::Cmp(
                        result_ty.clone(),
                        qbe::Cmp::Eq,
                        operand,
                        qbe::Value::Const(0),
                    ),
                    TypeKind::I64 => qbe::Instr::Xor(operand, qbe::Value::Const(u64::MAX)),
                    _ => unreachable!(),
                }
            }
            UnaryOp::Ref => {
                let addr = self.address_of(qfunc, operand_expr)?;
                return Ok(match addr {
                    qbe::Value::Temporary(name) => GenValue::Temp(name, expr.type_id),
                    qbe::Value::Global(name) => GenValue::Global(name, expr.type_id),
                    qbe::Value::Const(_) => unreachable!(),
                });
            }
            UnaryOp::Deref => {
                let ptr = self.generate_expression(qfunc, operand_expr)?.into();
                return Ok(self.load_value(qfunc, ptr, expr.type_id));
            }
        };

        Ok(self.assign_to_temp(qfunc, expr.type_id, instr))
    }

    fn generate_expr_binary(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(binop, expr1, expr2) = &expr.kind else {
            unreachable!()
        };

        match binop {
            BinaryOp::And => self.generate_expr_land(qfunc, expr),
            BinaryOp::Or => self.generate_expr_lor(qfunc, expr),
            _ => {
                let operand1 = self.generate_expression(qfunc, expr1)?.into();
                let operand2 = self.generate_expression(qfunc, expr2)?.into();

                let instr = match binop {
                    BinaryOp::Add => qbe::Instr::Add(operand1, operand2),
                    BinaryOp::Sub => qbe::Instr::Sub(operand1, operand2),
                    BinaryOp::Mul => qbe::Instr::Mul(operand1, operand2),
                    BinaryOp::Div => qbe::Instr::Div(operand1, operand2),
                    BinaryOp::Rem => qbe::Instr::Rem(operand1, operand2),

                    BinaryOp::BitAnd => qbe::Instr::And(operand1, operand2),
                    BinaryOp::BitOr => qbe::Instr::Or(operand1, operand2),

                    cmp => {
                        let operand_ty = self.qbe_type(expr1.type_id);
                        qbe::Instr::Cmp(
                            operand_ty,
                            match cmp {
                                BinaryOp::Lt => qbe::Cmp::Slt,
                                BinaryOp::Le => qbe::Cmp::Sle,
                                BinaryOp::Gt => qbe::Cmp::Sgt,
                                BinaryOp::Ge => qbe::Cmp::Sge,
                                BinaryOp::Eq => qbe::Cmp::Eq,
                                BinaryOp::Ne => qbe::Cmp::Ne,
                                _ => unreachable!(),
                            },
                            operand1,
                            operand2,
                        )
                    }
                };

                Ok(self.assign_to_temp(qfunc, expr.type_id, instr))
            }
        }
    }

    fn generate_expr_assign(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let addr = self.address_of(qfunc, lhs)?;
        let value = self.generate_expression(qfunc, rhs)?;
        self.store_value(qfunc, addr, value, rhs.type_id);

        Ok(GenValue::Const(0, expr.type_id))
    }

    fn generate_expr_return(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Return(ret_expr) = &expr.kind else {
            unreachable!()
        };

        let value = ret_expr
            .as_ref()
            .map(|e| self.generate_expression(qfunc, e))
            .transpose()?
            .map(GenValue::into);

        qfunc.add_instr(qbe::Instr::Ret(value));

        Ok(GenValue::Const(0, expr.type_id))
    }

    fn generate_expr_control(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        match expr.kind {
            ExprKind::Break => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("break outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.break_label.clone()));
                Ok(GenValue::Const(0, expr.type_id))
            }
            ExprKind::Continue => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("continue outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.continue_label.clone()));
                Ok(GenValue::Const(0, expr.type_id))
            }

            _ => unreachable!(),
        }
    }

    fn generate_expr_array(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Array(elements) = &expr.kind else {
            unreachable!()
        };

        let TypeKind::Array(elem_ty, _) = self.types.get(expr.type_id).kind else {
            return Err(CodeGenError::new(
                format!("Expected array type, found {:?}", expr.type_id),
                expr.span,
            ));
        };

        let elem_type = self.types.get(elem_ty);
        let elem_size = elem_type.size as u64;
        let array_size = elements.len() as u64 * elem_size;
        let array_name = self.new_temp();
        let array_ptr = qbe::Value::Temporary(array_name.clone());
        qfunc.assign_instr(
            array_ptr.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        for (i, elem) in elements.iter().enumerate() {
            let elem_val: qbe::Value = self.generate_expression(qfunc, elem)?.into();
            let offset = i as u64 * elem_size;
            self.store_field(qfunc, array_ptr.clone(), offset, elem_val, elem_ty);
        }

        Ok(GenValue::Temp(array_name, expr.type_id))
    }

    fn generate_expr_repeat(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Repeat(elem, Literal::Integer(count)) = &expr.kind else {
            unreachable!()
        };

        let TypeKind::Array(elem_ty, _) = self.types.get(expr.type_id).kind else {
            return Err(CodeGenError::new(
                format!("Expected array type, found {:?}", expr.type_id),
                expr.span,
            ));
        };

        let elem_type = self.types.get(elem_ty);
        let elem_size = elem_type.size as u64;
        let array_size = *count as u64 * elem_size;
        let array_name = self.new_temp();
        let array_ptr = qbe::Value::Temporary(array_name.clone());
        qfunc.assign_instr(
            array_ptr.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        let elem_val: qbe::Value = self.generate_expression(qfunc, elem)?.into();

        for i in 0..*count {
            let offset = i as u64 * elem_size;
            self.store_field(qfunc, array_ptr.clone(), offset, elem_val.clone(), elem_ty);
        }

        Ok(GenValue::Temp(array_name, expr.type_id))
    }

    fn generate_expr_index(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Index(..) = &expr.kind else {
            unreachable!()
        };

        let addr = self.address_of(qfunc, expr)?;
        Ok(self.load_value(qfunc, addr, expr.type_id))
    }

    fn generate_expr_field(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Field(..) = &expr.kind else {
            unreachable!()
        };

        let addr = self.address_of(qfunc, expr)?;
        Ok(self.load_value(qfunc, addr, expr.type_id))
    }

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let result = match &expr.kind {
            ExprKind::Literal(..) => Ok(self.generate_expr_literal(qfunc, expr)),
            ExprKind::Place(..) => Ok(self.generate_expr_place(qfunc, expr)),
            ExprKind::Struct(..) => self.generate_expr_struct(qfunc, expr),
            ExprKind::Call(..) => self.generate_expr_call(qfunc, expr),
            ExprKind::Unary(..) => self.generate_expr_unary(qfunc, expr),
            ExprKind::Binary(..) => self.generate_expr_binary(qfunc, expr),
            ExprKind::Assign(..) => self.generate_expr_assign(qfunc, expr),
            ExprKind::Return(..) => self.generate_expr_return(qfunc, expr),
            ExprKind::Block(block) => self.generate_block(qfunc, block),
            ExprKind::If(..) => self.generate_expr_if(qfunc, expr),
            ExprKind::While(..) => self.generate_expr_while(qfunc, expr),
            ExprKind::Break | ExprKind::Continue => self.generate_expr_control(qfunc, expr),
            ExprKind::Array(..) => self.generate_expr_array(qfunc, expr),
            ExprKind::Repeat(..) => self.generate_expr_repeat(qfunc, expr),
            ExprKind::Index(..) => self.generate_expr_index(qfunc, expr),
            ExprKind::Field(..) => self.generate_expr_field(qfunc, expr),
        }?;

        if expr.type_id == TypeId::NEVER {
            let cont_label = format!("never.{}", self.new_label());
            qfunc.add_block(cont_label);
        }

        Ok(result)
    }

    fn emit_string_data(&mut self, s: &str) -> String {
        let label = format!("str.{}", self.string_counter);
        self.string_counter += 1;

        self.data_defs.push(qbe::DataDef::new(
            qbe::Linkage::private(),
            label.clone(),
            None,
            vec![(qbe::Type::Byte, qbe::DataItem::Str(s.to_string()))],
        ));

        label
    }

    fn emit_cstring_data(&mut self, s: &str) -> String {
        let label = format!("cstr.{}", self.string_counter);
        self.string_counter += 1;

        self.data_defs.push(qbe::DataDef::new(
            qbe::Linkage::private(),
            label.clone(),
            None,
            vec![
                (qbe::Type::Byte, qbe::DataItem::Str(s.to_string())),
                (qbe::Type::Byte, qbe::DataItem::Const(0)),
            ],
        ));

        label
    }

    fn generate_string_slice(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> GenValue {
        let ExprKind::Literal(Literal::String(s)) = &expr.kind else {
            unreachable!()
        };

        // Emit global string data
        let data_label = self.emit_string_data(s);

        // Allocate slice on stack: { ptr: l, len: l }
        let slice_addr = self.alloc_local(qfunc, &Type::slice(TypeId::U8));

        // Store ptr field
        let ptr = qbe::Value::Global(data_label);
        let ptr_type_id = self.types.mk_pointer(TypeId::U8);
        self.store_field(qfunc, slice_addr.clone(), 0, ptr, ptr_type_id);

        // Store len field
        let len = qbe::Value::Const(s.len() as u64);
        self.store_field(qfunc, slice_addr.clone(), 8, len, TypeId::I64);

        // Return address of slice
        match slice_addr {
            qbe::Value::Temporary(name) => GenValue::Temp(name, expr.type_id),
            _ => unreachable!(),
        }
    }

    fn generate_expr_call(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Call(call) = &expr.kind else {
            unreachable!()
        };

        let symbol = match &call.callee.kind {
            ExprKind::Place(Place::Local(ident) | Place::Global(ident)) => {
                Self::ident_to_symbol(ident)
            }
            _ => {
                return Err(CodeGenError::new(
                    "callee must be an identifier".to_string(),
                    call.callee.span,
                ));
            }
        };

        let mut qbe_args = Vec::new();
        for arg in &call.args {
            let arg_val = self.generate_expression(qfunc, arg)?.into();
            let arg_ty = self.qbe_type(arg.type_id);
            qbe_args.push((arg_ty, arg_val));
        }

        if expr.type_id == TypeId::UNIT {
            qfunc.add_instr(qbe::Instr::Call(symbol, qbe_args, None));
            Ok(GenValue::Const(0, expr.type_id))
        } else {
            Ok(self.assign_to_temp(
                qfunc,
                expr.type_id,
                qbe::Instr::Call(symbol, qbe_args, None),
            ))
        }
    }

    fn generate_expr_if(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let label_id = self.new_label();
        let cond_label = format!("if.{label_id}.cond");
        let then_label = format!("if.{label_id}.then");
        let end_label = format!("if.{label_id}.end");

        qfunc.add_block(cond_label);
        let cond = self.generate_expression(qfunc, &if_expr.cond)?.into();

        match &if_expr.else_body {
            None => {
                // if without else: always returns Unit
                qfunc.add_instr(qbe::Instr::Jnz(cond, then_label.clone(), end_label.clone()));

                qfunc.add_block(then_label);
                self.generate_block(qfunc, &if_expr.then_body)?;
                if if_expr.then_body.type_id != TypeId::NEVER {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                qfunc.add_block(end_label);
                Ok(GenValue::Const(0, expr.type_id))
            }
            Some(else_expr) => {
                let else_label = format!("if.{label_id}.else");
                qfunc.add_instr(qbe::Instr::Jnz(
                    cond,
                    then_label.clone(),
                    else_label.clone(),
                ));

                // Generate then branch
                qfunc.add_block(then_label.clone());
                let then_result = self.generate_block(qfunc, &if_expr.then_body)?;
                let then_predecessor = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty")
                    .label
                    .clone();
                let then_is_never = if_expr.then_body.type_id == TypeId::NEVER;
                if !then_is_never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // Generate else branch
                qfunc.add_block(else_label.clone());
                let else_result = self.generate_expression(qfunc, else_expr)?;
                let else_predecessor = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty")
                    .label
                    .clone();
                let else_is_never = else_expr.type_id == TypeId::NEVER;
                if !else_is_never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // Generate merge block
                qfunc.add_block(end_label);

                // Determine result based on expression type and branch types
                match expr.type_id {
                    TypeId::UNIT | TypeId::NEVER => Ok(GenValue::Const(0, expr.type_id)),
                    _ => match (then_is_never, else_is_never) {
                        (true, false) => {
                            // Only else branch has value
                            Ok(else_result)
                        }
                        (false, true) => {
                            // Only then branch has value
                            Ok(then_result)
                        }
                        (false, false) => {
                            let then_val = then_result.into();
                            let else_val = else_result.into();
                            Ok(self.assign_to_temp(
                                qfunc,
                                expr.type_id,
                                qbe::Instr::Phi(
                                    then_predecessor,
                                    then_val,
                                    else_predecessor,
                                    else_val,
                                ),
                            ))
                        }
                        (true, true) => unreachable!(),
                    },
                }
            }
        }
    }

    fn generate_expr_while(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        let label_id = self.new_label();
        let cond_label = format!("while.{label_id}.cond");
        let body_label = format!("while.{label_id}.body");
        let end_label = format!("while.{label_id}.end");

        qfunc.add_block(cond_label.clone());
        let cond_val = self.generate_expression(qfunc, &while_expr.cond)?.into();
        qfunc.add_instr(qbe::Instr::Jnz(
            cond_val,
            body_label.clone(),
            end_label.clone(),
        ));

        // Body block
        self.push_loop(cond_label.clone(), end_label.clone());

        qfunc.add_block(body_label);
        self.generate_block(qfunc, &while_expr.body)?;

        if !qfunc.blocks.last().is_some_and(qbe::Block::jumps) {
            qfunc.add_instr(qbe::Instr::Jmp(cond_label));
        }

        self.pop_loop();

        // End block
        qfunc.add_block(end_label);

        Ok(GenValue::Const(0, expr.type_id))
    }

    fn generate_expr_land(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(BinaryOp::And, left, right) = &expr.kind else {
            unreachable!()
        };

        let result_ty = self.qbe_type(expr.type_id);
        let left_val = self.generate_expression(qfunc, left)?.into();

        let label_id = self.new_label();
        let rhs_label = format!("land.{label_id}.rhs");
        let false_label = format!("land.{label_id}.false");
        let end_label = format!("land.{label_id}.end");

        qfunc.add_instr(qbe::Instr::Jnz(
            left_val,
            rhs_label.clone(),
            false_label.clone(),
        ));

        qfunc.add_block(rhs_label.clone());
        let right_val = self.generate_expression(qfunc, right)?.into();
        let right_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            right_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(right_val),
        );
        let rhs_predecessor = qfunc
            .blocks
            .last()
            .expect("ICE: blocks should not be empty")
            .label
            .clone();

        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(false_label.clone());
        let false_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            false_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(qbe::Value::Const(0)),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(end_label);
        Ok(self.assign_to_temp(
            qfunc,
            expr.type_id,
            qbe::Instr::Phi(rhs_predecessor, right_temp, false_label, false_temp),
        ))
    }

    fn generate_expr_lor(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(BinaryOp::Or, left, right) = &expr.kind else {
            unreachable!()
        };

        let result_ty = self.qbe_type(expr.type_id);
        let left_val = self.generate_expression(qfunc, left)?.into();

        let label_id = self.new_label();
        let rhs_label = format!("lor.{label_id}.rhs");
        let true_label = format!("lor.{label_id}.true");
        let end_label = format!("lor.{label_id}.end");

        qfunc.add_instr(qbe::Instr::Jnz(
            left_val,
            true_label.clone(),
            rhs_label.clone(),
        ));

        qfunc.add_block(rhs_label.clone());
        let right_val = self.generate_expression(qfunc, right)?.into();
        let right_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            right_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(right_val),
        );
        let rhs_predecessor = qfunc
            .blocks
            .last()
            .expect("ICE: blocks should not be empty")
            .label
            .clone();

        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(true_label.clone());
        let true_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            true_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(qbe::Value::Const(1)),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(end_label);
        Ok(self.assign_to_temp(
            qfunc,
            expr.type_id,
            qbe::Instr::Phi(rhs_predecessor, right_temp, true_label, true_temp),
        ))
    }
}
