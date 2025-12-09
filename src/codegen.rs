use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use crate::{ast::*, span::Span};

#[derive(Debug, Clone)]
pub enum GenValue {
    Const(u64, Type),
    Global(String, Type),
    Temp(String, Type),
}

impl GenValue {
    pub fn ty(&self) -> &Type {
        match self {
            GenValue::Const(_, ty) | GenValue::Global(_, ty) | GenValue::Temp(_, ty) => ty,
        }
    }

    pub fn into_qbe(self) -> qbe::Value {
        self.into()
    }
}

impl From<GenValue> for qbe::Value {
    fn from(value: GenValue) -> Self {
        match value.ty() {
            Type::Unit | Type::Never => unreachable!(),
            _ => match value {
                GenValue::Const(val, _) => qbe::Value::Const(val),
                GenValue::Global(name, _) => qbe::Value::Global(name),
                GenValue::Temp(name, _) => qbe::Value::Temporary(name),
            },
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

#[derive(Default)]
pub struct CodeGen {
    temp_counter: usize,
    label_counter: usize,
    string_counter: usize,
    scopes: Vec<HashSet<String>>,
    loops: Vec<LoopContext>,
    constants: HashMap<String, Literal>,
    data_defs: Vec<qbe::DataDef<'static>>,
    globals: HashSet<String>,
}

impl CodeGen {
    pub fn new() -> Self {
        Self {
            temp_counter: 0,
            label_counter: 0,
            string_counter: 0,
            scopes: Vec::new(),
            loops: Vec::new(),
            constants: HashMap::new(),
            data_defs: Vec::new(),
            globals: HashSet::new(),
        }
    }

    pub fn generate(&mut self, prog: &Program<Type>) -> Result<String, CodeGenError> {
        let mut module = qbe::Module::new();

        // Constants
        for item in &prog.items {
            if let Item::Const(const_def) = item {
                let value = self.eval_const_expr(&const_def.init)?;
                self.constants.insert(const_def.name.clone(), value);
            }
        }

        // Globals
        for item in &prog.items {
            match item {
                Item::Static(static_def) => {
                    self.generate_static(static_def)?;
                    self.globals.insert(static_def.name.clone());
                }
                Item::Extern(ExternItem::Static(static_decl)) => {
                    self.globals.insert(static_decl.name.clone());
                }
                _ => {}
            }
        }

        // Functions
        for item in &prog.items {
            if let Item::Function(func) = item {
                module.add_function(self.generate_function(func)?);
            }
        }

        // DataDefs
        for data_def in &self.data_defs {
            module.add_data(data_def.clone());
        }

        Ok(module.to_string())
    }

    fn new_temp(&mut self) -> String {
        let name = format!("temp.{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn assign_to_temp(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        ty: &Type,
        instr: qbe::Instr<'static>,
    ) -> GenValue {
        let name = self.new_temp();
        let temp = qbe::Value::Temporary(name.clone());
        let qbe_ty = qbe::Type::from(ty);
        qfunc.assign_instr(temp, qbe_ty, instr);
        GenValue::Temp(name, ty.clone())
    }

    fn new_label(&mut self) -> usize {
        let id = self.label_counter;
        self.label_counter += 1;
        id
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashSet::new());
    }

    fn pop_scope(&mut self) {
        self.scopes
            .pop()
            .expect("ICE: cannot pop scope, scopes stack is empty");
    }

    fn is_constant(&self, name: &str) -> bool {
        self.constants.contains_key(name)
    }

    fn is_global_var(&self, name: &str) -> bool {
        self.globals.contains(name)
    }

    fn is_local_var(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|scope| scope.contains(name))
    }

    fn insert_local_var(&mut self, name: String) {
        self.scopes
            .last_mut()
            .expect("ICE: scopes stack should not be empty")
            .insert(name);
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
        expr: &Expr<Type>,
    ) -> Result<qbe::Value, CodeGenError> {
        match &expr.kind {
            // x -> %x
            ExprKind::Ident(name) => {
                // Constants
                if self.is_constant(name) {
                    return Err(CodeGenError::new(
                        format!("Cannot take address of constant `{name}`"),
                        expr.span,
                    ));
                }

                // Global variables
                if self.is_global_var(name) {
                    return Ok(qbe::Value::Global(name.clone()));
                }

                // Local variables
                if self.is_local_var(name) {
                    return Ok(qbe::Value::Temporary(name.clone()));
                }

                Err(CodeGenError::new(
                    format!("Cannot find value `{name}` in this scope"),
                    expr.span,
                ))
            }
            // *ptr -> value_of(ptr)
            ExprKind::Unary(UnaryOp::Deref, ptr_expr) => {
                Ok(self.generate_expression(qfunc, ptr_expr)?.into_qbe())
            }
            // arr[i] -> value_of(arr) + i * elem_size
            ExprKind::Index(base, index) => {
                let base_ptr = self.generate_expression(qfunc, base)?.into_qbe();
                let index_val = self.generate_expression(qfunc, index)?.into_qbe();

                let offset = qbe::Value::Temporary(self.new_temp());
                qfunc.assign_instr(
                    offset.clone(),
                    qbe::Type::Long,
                    qbe::Instr::Mul(
                        index_val,
                        qbe::Value::Const(qbe::Type::from(&base.ty).size()),
                    ),
                );

                let addr = qbe::Value::Temporary(self.new_temp());
                qfunc.assign_instr(
                    addr.clone(),
                    qbe::Type::Long,
                    qbe::Instr::Add(base_ptr, offset),
                );

                Ok(addr)
            }
            _ => Err(CodeGenError::new(
                "Invalid left-hand side of assignment".to_string(),
                expr.span,
            )),
        }
    }

    fn literal_to_data_item(&mut self, lit: &Literal) -> qbe::DataItem {
        match lit {
            Literal::Integer(n) => qbe::DataItem::Const(n.cast_unsigned()),
            Literal::Bool(b) => qbe::DataItem::Const(u64::from(*b)),
            Literal::String(s) => {
                let label = self.emit_string_data(s);
                qbe::DataItem::Symbol(label, None)
            }
        }
    }

    fn eval_const_expr(&mut self, expr: &Expr<Type>) -> Result<Literal, CodeGenError> {
        match &expr.kind {
            ExprKind::Literal(lit) => Ok(lit.clone()),
            ExprKind::Ident(name) => self.constants.get(name).cloned().ok_or_else(|| {
                CodeGenError::new(format!("Constant `{name}` not found"), expr.span)
            }),
            ExprKind::Unary(UnaryOp::Neg, operand) => match self.eval_const_expr(operand)? {
                Literal::Integer(n) => Ok(Literal::Integer(-n)),
                _ => Err(CodeGenError::new(
                    "Cannot negate a non-integer value".to_string(),
                    expr.span,
                )),
            },
            ExprKind::Binary(op, left, right) => {
                let left_val = self.eval_const_expr(left)?;
                let right_val = self.eval_const_expr(right)?;

                match (left_val, right_val) {
                    (Literal::Integer(l), Literal::Integer(r)) => match op {
                        // Arithmetic operators
                        BinaryOp::Add => Ok(Literal::Integer(l + r)),
                        BinaryOp::Sub => Ok(Literal::Integer(l - r)),
                        BinaryOp::Mul => Ok(Literal::Integer(l * r)),
                        BinaryOp::Div => Ok(Literal::Integer(l / r)),
                        BinaryOp::Rem => Ok(Literal::Integer(l % r)),
                        // Bitwise operators
                        BinaryOp::BitAnd => Ok(Literal::Integer(l & r)),
                        BinaryOp::BitOr => Ok(Literal::Integer(l | r)),
                        // Comparison operators
                        BinaryOp::Lt => Ok(Literal::Bool(l < r)),
                        BinaryOp::Le => Ok(Literal::Bool(l <= r)),
                        BinaryOp::Gt => Ok(Literal::Bool(l > r)),
                        BinaryOp::Ge => Ok(Literal::Bool(l >= r)),
                        BinaryOp::Eq => Ok(Literal::Bool(l == r)),
                        BinaryOp::Ne => Ok(Literal::Bool(l != r)),
                        _ => Err(CodeGenError::new(
                            "Invalid operator for integer operands".to_string(),
                            expr.span,
                        )),
                    },
                    (Literal::Bool(l), Literal::Bool(r)) => match op {
                        // Logical operators
                        BinaryOp::And => Ok(Literal::Bool(l && r)),
                        BinaryOp::Or => Ok(Literal::Bool(l || r)),
                        // Equality operators
                        BinaryOp::Eq => Ok(Literal::Bool(l == r)),
                        BinaryOp::Ne => Ok(Literal::Bool(l != r)),
                        _ => Err(CodeGenError::new(
                            "Invalid operator for boolean operands".to_string(),
                            expr.span,
                        )),
                    },
                    _ => Err(CodeGenError::new(
                        "Type mismatch in constant expression".to_string(),
                        expr.span,
                    )),
                }
            }
            _ => Err(CodeGenError::new(
                "Invalid constant expression".to_string(),
                expr.span,
            )),
        }
    }

    fn generate_static(&mut self, static_def: &StaticDef<Type>) -> Result<(), CodeGenError> {
        let literal = self.eval_const_expr(&static_def.init)?;
        let qbe_ty = qbe::Type::from(&static_def.type_ann);
        let data_item = self.literal_to_data_item(&literal);

        self.data_defs.push(qbe::DataDef::new(
            qbe::Linkage::private(),
            static_def.name.clone(),
            None,
            vec![(qbe_ty, data_item)],
        ));

        Ok(())
    }

    fn generate_function(
        &mut self,
        func: &FunctionDef<Type>,
    ) -> Result<qbe::Function<'static>, CodeGenError> {
        self.push_scope();

        let params = func
            .params
            .iter()
            .map(|param| {
                let ty = qbe::Type::from(&param.type_ann);
                let value = qbe::Value::Temporary(format!("{}.param", param.name));
                (ty, value)
            })
            .collect();

        let qbe_return_type = if func.return_type_ann == Type::Unit {
            None
        } else {
            Some(qbe::Type::from(&func.return_type_ann))
        };

        let mut qfunc = qbe::Function::new(
            qbe::Linkage::public(),
            func.name.clone(),
            params,
            qbe_return_type,
        );

        qfunc.add_block("start");

        for param in &func.params {
            let ty = qbe::Type::from(&param.type_ann);
            let addr = qbe::Value::Temporary(param.name.clone());
            let param_val = qbe::Value::Temporary(format!("{}.param", param.name));

            qfunc.assign_instr(addr.clone(), qbe::Type::Long, qbe::Instr::Alloc8(8));
            qfunc.add_instr(qbe::Instr::Store(ty, addr, param_val));

            self.insert_local_var(param.name.clone());
        }

        qfunc.add_block("body");

        let block_value = self.generate_block(&mut qfunc, &func.body)?;

        if func.return_type_ann == Type::Never {
            qfunc.add_instr(qbe::Instr::Hlt);
        } else if func.body.ty == Type::Never {
            if let Some(last_block) = qfunc.blocks.last()
                && !last_block.jumps()
            {
                qfunc.add_instr(qbe::Instr::Hlt);
            }
        } else if func.body.ty == Type::Unit {
            qfunc.add_instr(qbe::Instr::Ret(None));
        } else {
            qfunc.add_instr(qbe::Instr::Ret(Some(block_value.into_qbe())));
        }

        self.pop_scope();

        Ok(qfunc)
    }

    fn generate_block(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        block: &Block<Type>,
    ) -> Result<GenValue, CodeGenError> {
        self.push_scope();
        let mut result = GenValue::Const(0, Type::Unit);
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
        self.pop_scope();
        Ok(result)
    }

    fn generate_let(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        let_stmt: &Let<Type>,
    ) -> Result<(), CodeGenError> {
        let qbe_ty = qbe::Type::from(&let_stmt.type_ann);
        let size = qbe_ty.size();

        let addr = qbe::Value::Temporary(let_stmt.name.clone());
        qfunc.assign_instr(addr.clone(), qbe::Type::Long, qbe::Instr::Alloc8(size));

        self.insert_local_var(let_stmt.name.clone());

        let init_val = self.generate_expression(qfunc, &let_stmt.init)?.into_qbe();
        qfunc.add_instr(qbe::Instr::Store(qbe_ty, addr, init_val));

        Ok(())
    }

    fn generate_expr_literal(&mut self, expr: &Expr<Type>) -> GenValue {
        let ExprKind::Literal(lit) = &expr.kind else {
            unreachable!()
        };

        match lit {
            Literal::Integer(n) => GenValue::Const(n.cast_unsigned(), Type::I64),
            Literal::String(s) => {
                let label = self.emit_string_data(s);
                GenValue::Global(label, Type::Str)
            }
            Literal::Bool(b) => GenValue::Const(u64::from(*b), Type::Bool),
        }
    }

    fn generate_expr_ident(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Ident(name) = &expr.kind else {
            unreachable!()
        };

        // Constants
        if let Some(literal) = self.constants.get(name).cloned() {
            return match literal {
                Literal::Integer(n) => Ok(GenValue::Const(n.cast_unsigned(), Type::I64)),
                Literal::Bool(b) => Ok(GenValue::Const(u64::from(b), Type::Bool)),
                Literal::String(s) => {
                    let label = self.emit_string_data(&s);
                    Ok(GenValue::Global(label, Type::Str))
                }
            };
        }

        // Local variables
        if self.is_local_var(name) {
            let addr = qbe::Value::Temporary(name.clone());
            let qbe_ty = qbe::Type::from(&expr.ty);
            return Ok(self.assign_to_temp(qfunc, &expr.ty, qbe::Instr::Load(qbe_ty, addr)));
        }

        // Global variables
        if self.is_global_var(name) {
            let addr = qbe::Value::Global(name.clone());
            let qbe_ty = qbe::Type::from(&expr.ty);
            return Ok(self.assign_to_temp(qfunc, &expr.ty, qbe::Instr::Load(qbe_ty, addr)));
        }

        Err(CodeGenError::new(
            format!("Cannot find value `{name}` in this scope"),
            expr.span,
        ))
    }

    fn generate_expr_unary(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Unary(unop, operand_expr) = &expr.kind else {
            unreachable!()
        };

        let instr = match unop {
            UnaryOp::Neg => {
                let operand = self.generate_expression(qfunc, operand_expr)?.into_qbe();
                qbe::Instr::Neg(operand)
            }
            UnaryOp::Not => {
                let operand = self.generate_expression(qfunc, operand_expr)?.into_qbe();
                let result_ty = qbe::Type::from(&expr.ty);
                match &operand_expr.ty {
                    Type::Bool => qbe::Instr::Cmp(
                        result_ty.clone(),
                        qbe::Cmp::Eq,
                        operand,
                        qbe::Value::Const(0),
                    ),
                    Type::I64 => qbe::Instr::Xor(operand, qbe::Value::Const(u64::MAX)),
                    _ => unreachable!(),
                }
            }
            UnaryOp::Ref => {
                let addr = self.address_of(qfunc, operand_expr)?;
                return Ok(match addr {
                    qbe::Value::Temporary(name) => GenValue::Temp(name, expr.ty.clone()),
                    qbe::Value::Global(name) => GenValue::Global(name, expr.ty.clone()),
                    qbe::Value::Const(_) => unreachable!(),
                });
            }
            UnaryOp::Deref => {
                let ptr = self.generate_expression(qfunc, operand_expr)?.into_qbe();
                let result_ty = qbe::Type::from(&expr.ty);
                qbe::Instr::Load(result_ty, ptr)
            }
        };

        Ok(self.assign_to_temp(qfunc, &expr.ty, instr))
    }

    fn generate_expr_binary(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(binop, expr1, expr2) = &expr.kind else {
            unreachable!()
        };

        match binop {
            BinaryOp::And => self.generate_expr_land(qfunc, expr),
            BinaryOp::Or => self.generate_expr_lor(qfunc, expr),
            _ => {
                let operand1 = self.generate_expression(qfunc, expr1)?.into_qbe();
                let operand2 = self.generate_expression(qfunc, expr2)?.into_qbe();

                let instr = match binop {
                    BinaryOp::Add => qbe::Instr::Add(operand1, operand2),
                    BinaryOp::Sub => qbe::Instr::Sub(operand1, operand2),
                    BinaryOp::Mul => qbe::Instr::Mul(operand1, operand2),
                    BinaryOp::Div => qbe::Instr::Div(operand1, operand2),
                    BinaryOp::Rem => qbe::Instr::Rem(operand1, operand2),

                    BinaryOp::BitAnd => qbe::Instr::And(operand1, operand2),
                    BinaryOp::BitOr => qbe::Instr::Or(operand1, operand2),

                    cmp => {
                        let operand_ty = qbe::Type::from(&expr1.ty);
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

                Ok(self.assign_to_temp(qfunc, &expr.ty, instr))
            }
        }
    }

    fn generate_expr_assign(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!()
        };

        let addr = self.address_of(qfunc, lhs)?;
        let value = self.generate_expression(qfunc, rhs)?.into_qbe();
        let qbe_ty = qbe::Type::from(&rhs.ty);
        qfunc.add_instr(qbe::Instr::Store(qbe_ty, addr, value));

        Ok(GenValue::Const(0, expr.ty.clone()))
    }

    fn generate_expr_return(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Return(ret_expr) = &expr.kind else {
            unreachable!()
        };

        let value = ret_expr
            .as_ref()
            .map(|e| self.generate_expression(qfunc, e))
            .transpose()?
            .map(GenValue::into_qbe);

        qfunc.add_instr(qbe::Instr::Ret(value));

        Ok(GenValue::Const(0, expr.ty.clone()))
    }

    fn generate_expr_control(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        match expr.kind {
            ExprKind::Break => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("break outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.break_label.clone()));
                Ok(GenValue::Const(0, expr.ty.clone()))
            }
            ExprKind::Continue => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("continue outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.continue_label.clone()));
                Ok(GenValue::Const(0, expr.ty.clone()))
            }

            _ => unreachable!(),
        }
    }

    fn generate_expr_array(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Array(elements) = &expr.kind else {
            unreachable!()
        };

        let elem_ty = match &expr.ty {
            Type::Array(elem_ty, _) => elem_ty.as_ref(),
            _ => {
                return Err(CodeGenError::new(
                    format!("Expected array type, found {:?}", expr.ty),
                    expr.span,
                ));
            }
        };

        let elem_size = qbe::Type::from(elem_ty).size();
        let array_size = elements.len() as u64 * elem_size;
        let array_name = self.new_temp();
        let array_ptr = qbe::Value::Temporary(array_name.clone());
        qfunc.assign_instr(
            array_ptr.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        let elem_qbe_ty = qbe::Type::from(elem_ty);
        for (i, elem) in elements.iter().enumerate() {
            let elem_val = self.generate_expression(qfunc, elem)?.into_qbe();
            let offset = i as u64 * elem_size;
            let elem_addr = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                elem_addr.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(array_ptr.clone(), qbe::Value::Const(offset)),
            );
            qfunc.add_instr(qbe::Instr::Store(elem_qbe_ty.clone(), elem_addr, elem_val));
        }

        Ok(GenValue::Temp(array_name, expr.ty.clone()))
    }

    fn generate_expr_repeat(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!()
        };

        let count_num = match self.eval_const_expr(count)? {
            Literal::Integer(n) => n as usize,
            _ => {
                return Err(CodeGenError::new(
                    "Expected integer constant".to_string(),
                    expr.span,
                ));
            }
        };

        let elem_ty = match &expr.ty {
            Type::Array(elem_ty, _) => elem_ty.as_ref(),
            _ => {
                return Err(CodeGenError::new(
                    format!("Expected array type, found {:?}", expr.ty),
                    expr.span,
                ));
            }
        };

        let elem_size = qbe::Type::from(elem_ty).size();
        let array_size = count_num as u64 * elem_size;
        let array_name = self.new_temp();
        let array_ptr = qbe::Value::Temporary(array_name.clone());
        qfunc.assign_instr(
            array_ptr.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        let elem_val = self.generate_expression(qfunc, elem)?.into_qbe();
        let elem_qbe_ty = qbe::Type::from(elem_ty);

        for i in 0..count_num {
            let offset = i as u64 * elem_size;
            let elem_addr = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                elem_addr.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(array_ptr.clone(), qbe::Value::Const(offset)),
            );
            qfunc.add_instr(qbe::Instr::Store(
                elem_qbe_ty.clone(),
                elem_addr,
                elem_val.clone(),
            ));
        }

        Ok(GenValue::Temp(array_name, expr.ty.clone()))
    }

    fn generate_expr_index(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Index(..) = &expr.kind else {
            unreachable!()
        };

        let addr = self.address_of(qfunc, expr)?;
        let qbe_ty = qbe::Type::from(&expr.ty);
        Ok(self.assign_to_temp(qfunc, &expr.ty, qbe::Instr::Load(qbe_ty, addr)))
    }

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let result = match &expr.kind {
            ExprKind::Literal(..) => Ok(self.generate_expr_literal(expr)),
            ExprKind::Ident(..) => self.generate_expr_ident(qfunc, expr),
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
        }?;

        if expr.ty == Type::Never {
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
            vec![
                (qbe::Type::Byte, qbe::DataItem::Str(s.to_string())),
                (qbe::Type::Byte, qbe::DataItem::Const(0)),
            ],
        ));

        label
    }

    fn generate_expr_call(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Call(call) = &expr.kind else {
            unreachable!()
        };

        let func_name = match &call.callee.kind {
            ExprKind::Ident(name) => name.clone(),
            _ => {
                return Err(CodeGenError::new(
                    "Complex callee expressions not yet supported".to_string(),
                    call.callee.span,
                ));
            }
        };

        let mut qbe_args = Vec::new();
        for arg in &call.args {
            let arg_val = self.generate_expression(qfunc, arg)?.into_qbe();
            let arg_ty = qbe::Type::from(&arg.ty);
            qbe_args.push((arg_ty, arg_val));
        }

        if call.callee.ty == Type::Unit {
            qfunc.add_instr(qbe::Instr::Call(func_name, qbe_args, None));
            Ok(GenValue::Const(0, expr.ty.clone()))
        } else {
            Ok(self.assign_to_temp(qfunc, &expr.ty, qbe::Instr::Call(func_name, qbe_args, None)))
        }
    }

    fn generate_expr_if(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::If(if_expr) = &expr.kind else {
            unreachable!()
        };

        let label_id = self.new_label();
        let cond_label = format!("if.{label_id}.cond");
        let then_label = format!("if.{label_id}.then");
        let end_label = format!("if.{label_id}.end");

        qfunc.add_block(cond_label);
        let cond = self.generate_expression(qfunc, &if_expr.cond)?.into_qbe();

        match &if_expr.else_body {
            None => {
                // if without else: always returns Unit
                qfunc.add_instr(qbe::Instr::Jnz(cond, then_label.clone(), end_label.clone()));

                qfunc.add_block(then_label);
                self.generate_block(qfunc, &if_expr.then_body)?;
                if if_expr.then_body.ty != Type::Never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                qfunc.add_block(end_label);
                Ok(GenValue::Const(0, expr.ty.clone()))
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
                let then_is_never = if_expr.then_body.ty == Type::Never;
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
                let else_is_never = else_expr.ty == Type::Never;
                if !else_is_never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // Generate merge block
                qfunc.add_block(end_label);

                // Determine result based on expression type and branch types
                match &expr.ty {
                    Type::Unit | Type::Never => Ok(GenValue::Const(0, expr.ty.clone())),
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
                            let then_val = then_result.into_qbe();
                            let else_val = else_result.into_qbe();
                            Ok(self.assign_to_temp(
                                qfunc,
                                &expr.ty,
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
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::While(while_expr) = &expr.kind else {
            unreachable!()
        };

        let label_id = self.new_label();
        let cond_label = format!("while.{label_id}.cond");
        let body_label = format!("while.{label_id}.body");
        let end_label = format!("while.{label_id}.end");

        qfunc.add_block(cond_label.clone());
        let cond_val = self
            .generate_expression(qfunc, &while_expr.cond)?
            .into_qbe();
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

        Ok(GenValue::Const(0, expr.ty.clone()))
    }

    fn generate_expr_land(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(BinaryOp::And, left, right) = &expr.kind else {
            unreachable!()
        };

        let result_ty = qbe::Type::from(&expr.ty);
        let left_val = self.generate_expression(qfunc, left)?.into_qbe();

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
        let right_val = self.generate_expression(qfunc, right)?.into_qbe();
        let right_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            right_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(right_val),
        );
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
            &expr.ty,
            qbe::Instr::Phi(rhs_label, right_temp, false_label, false_temp),
        ))
    }

    fn generate_expr_lor(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<GenValue, CodeGenError> {
        let ExprKind::Binary(BinaryOp::Or, left, right) = &expr.kind else {
            unreachable!()
        };

        let result_ty = qbe::Type::from(&expr.ty);
        let left_val = self.generate_expression(qfunc, left)?.into_qbe();

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
        let right_val = self.generate_expression(qfunc, right)?.into_qbe();
        let right_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            right_temp.clone(),
            result_ty.clone(),
            qbe::Instr::Copy(right_val),
        );
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
            &expr.ty,
            qbe::Instr::Phi(rhs_label, right_temp, true_label, true_temp),
        ))
    }
}
