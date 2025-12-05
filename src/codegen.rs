use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use crate::{ast::*, span::Span};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarKind {
    Param,
    Local,
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

pub struct CodeGen {
    temp_counter: usize,
    label_counter: usize,
    string_counter: usize,
    scopes: Vec<HashMap<String, VarKind>>,
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

        for item in &prog.items {
            if let Item::Const(const_def) = item {
                let value = self.eval_const_expr(&const_def.init)?;
                self.constants.insert(const_def.name.clone(), value);
            }
        }

        for item in &prog.items {
            if let Item::Static(static_def) = item {
                self.generate_static(static_def)?;
                self.globals.insert(static_def.name.clone());
            }
        }

        for item in &prog.items {
            if let Item::Function(func) = item
                && func.body.is_some()
            {
                module.add_function(self.generate_function(func)?);
            }
        }

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

    fn new_label(&mut self) -> usize {
        let id = self.label_counter;
        self.label_counter += 1;
        id
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes
            .pop()
            .expect("ICE: cannot pop scope, scopes stack is empty");
    }

    fn insert_local(&mut self, name: String) {
        self.scopes
            .last_mut()
            .expect("ICE: scopes stack should not be empty")
            .insert(name, VarKind::Local);
    }

    fn insert_param(&mut self, name: String) {
        self.scopes
            .last_mut()
            .expect("ICE: scopes stack should not be empty")
            .insert(name, VarKind::Param);
    }

    fn lookup_var(&self, name: &str) -> Option<VarKind> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
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
            ExprKind::Ident(name) => {
                if self.constants.contains_key(name) {
                    return Err(CodeGenError::new(
                        format!("Cannot assign to constant `{name}`"),
                        expr.span,
                    ));
                }

                if self.globals.contains(name) {
                    return Ok(qbe::Value::Global(name.clone()));
                }

                if self.lookup_var(name).is_some() {
                    return Ok(qbe::Value::Temporary(name.clone()));
                }

                Err(CodeGenError::new(
                    format!("Cannot find value `{name}` in this scope"),
                    expr.span,
                ))
            }
            ExprKind::Index(base, index) => {
                let base_val = self.generate_expression(qfunc, base)?.ok_or_else(|| {
                    CodeGenError::new(
                        "Array expression must produce a value".to_string(),
                        base.span,
                    )
                })?;

                let index_val = self.generate_expression(qfunc, index)?.ok_or_else(|| {
                    CodeGenError::new(
                        "Index expression must produce a value".to_string(),
                        index.span,
                    )
                })?;

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
                    qbe::Instr::Add(base_val, offset),
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
        let qbe_ty = qbe::Type::from(&Type::from(&static_def.type_ann));
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

        for param in &func.params {
            self.insert_param(param.name.clone());
        }

        let params = func
            .params
            .iter()
            .map(|param| {
                let ty = qbe::Type::from(&Type::from(&param.type_ann));
                let value = qbe::Value::Temporary(param.name.clone());
                (ty, value)
            })
            .collect();

        let return_type = Type::from(&func.return_type_ann);

        let qbe_return_type = if return_type == Type::Unit {
            None
        } else {
            Some(qbe::Type::from(&return_type))
        };

        let mut qfunc = qbe::Function::new(
            qbe::Linkage::public(),
            func.name.clone(),
            params,
            qbe_return_type,
        );

        let body = func
            .body
            .as_ref()
            .expect("generate_function called on prototype");

        qfunc.add_block("start");
        let block_value = self.generate_block(&mut qfunc, body)?;

        if return_type == Type::Never {
            qfunc.add_instr(qbe::Instr::Hlt);
        } else if body.ty == Type::Never {
            if let Some(last_block) = qfunc.blocks.last()
                && !last_block.jumps()
            {
                qfunc.add_instr(qbe::Instr::Hlt);
            }
        } else {
            qfunc.add_instr(qbe::Instr::Ret(block_value));
        }

        self.pop_scope();

        Ok(qfunc)
    }

    fn generate_block(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        block: &Block<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        self.push_scope();
        let mut result = None;
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
        let qbe_ty = qbe::Type::from(&Type::from(&let_stmt.type_ann));

        let size = qbe_ty.size();

        let addr = qbe::Value::Temporary(let_stmt.name.clone());
        qfunc.assign_instr(addr.clone(), qbe::Type::Long, qbe::Instr::Alloc8(size));

        self.insert_local(let_stmt.name.clone());

        let init_val = self
            .generate_expression(qfunc, &let_stmt.init)?
            .ok_or_else(|| {
                CodeGenError::new(
                    "Variable initialization requires a value".to_string(),
                    let_stmt.init.span,
                )
            })?;
        qfunc.add_instr(qbe::Instr::Store(qbe_ty, addr, init_val));

        Ok(())
    }

    fn generate_expr_literal(&mut self, expr: &Expr<Type>) -> qbe::Value {
        let ExprKind::Literal(lit) = &expr.kind else {
            unreachable!("ICE: generate_expr_literal called with non-Literal expression")
        };
        match lit {
            Literal::Integer(n) => qbe::Value::Const(n.cast_unsigned()),
            Literal::String(s) => {
                let label = self.emit_string_data(s);
                qbe::Value::Global(label)
            }
            Literal::Bool(b) => qbe::Value::Const(u64::from(*b)),
        }
    }

    fn generate_expr_ident(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Ident(name) = &expr.kind else {
            unreachable!("ICE: generate_expr_ident called with non-Ident expression")
        };

        if let Some(literal) = self.constants.get(name).cloned() {
            return match literal {
                Literal::Integer(n) => Ok(Some(qbe::Value::Const(n.cast_unsigned()))),
                Literal::Bool(b) => Ok(Some(qbe::Value::Const(u64::from(b)))),
                Literal::String(s) => {
                    let label = self.emit_string_data(&s);
                    Ok(Some(qbe::Value::Global(label)))
                }
            };
        }

        if let Some(var_kind) = self.lookup_var(name) {
            match var_kind {
                VarKind::Param => {
                    return Ok(Some(qbe::Value::Temporary(name.clone())));
                }
                VarKind::Local => {
                    let addr = qbe::Value::Temporary(name.clone());
                    let result = qbe::Value::Temporary(self.new_temp());
                    let qbe_ty = qbe::Type::from(&expr.ty);
                    qfunc.assign_instr(
                        result.clone(),
                        qbe_ty.clone(),
                        qbe::Instr::Load(qbe_ty, addr),
                    );
                    return Ok(Some(result));
                }
            }
        }

        if self.globals.contains(name) {
            let addr = qbe::Value::Global(name.clone());
            let result = qbe::Value::Temporary(self.new_temp());
            let qbe_ty = qbe::Type::from(&expr.ty);
            qfunc.assign_instr(
                result.clone(),
                qbe_ty.clone(),
                qbe::Instr::Load(qbe_ty, addr),
            );
            return Ok(Some(result));
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
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Unary(unop, operand_expr) = &expr.kind else {
            unreachable!("ICE: generate_expr_unary called with non-Unary expression")
        };

        let operand = self
            .generate_expression(qfunc, operand_expr)?
            .ok_or_else(|| {
                CodeGenError::new(
                    "Expression must produce a value".to_string(),
                    operand_expr.span,
                )
            })?;
        let result = qbe::Value::Temporary(self.new_temp());
        let result_ty = qbe::Type::from(&expr.ty);

        match unop {
            UnaryOp::Neg => {
                qfunc.assign_instr(result.clone(), result_ty, qbe::Instr::Neg(operand));
            }
            UnaryOp::Not => match &operand_expr.ty {
                Type::Bool => qfunc.assign_instr(
                    result.clone(),
                    result_ty.clone(),
                    qbe::Instr::Cmp(result_ty, qbe::Cmp::Eq, operand, qbe::Value::Const(0)),
                ),
                Type::I64 => qfunc.assign_instr(
                    result.clone(),
                    result_ty.clone(),
                    qbe::Instr::Xor(operand, qbe::Value::Const(u64::MAX)),
                ),
                _ => unreachable!(),
            },
        }

        Ok(Some(result))
    }

    fn generate_expr_binary(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Binary(binop, expr1, expr2) = &expr.kind else {
            unreachable!("ICE: generate_expr_binary called with non-Binary expression")
        };

        match binop {
            BinaryOp::And => self.generate_expr_land(qfunc, expr),
            BinaryOp::Or => self.generate_expr_lor(qfunc, expr),
            _ => {
                let operand1 = self.generate_expression(qfunc, expr1)?.ok_or_else(|| {
                    CodeGenError::new("Expression must produce a value".to_string(), expr1.span)
                })?;
                let operand2 = self.generate_expression(qfunc, expr2)?.ok_or_else(|| {
                    CodeGenError::new("Expression must produce a value".to_string(), expr2.span)
                })?;
                let result = qbe::Value::Temporary(self.new_temp());

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

                let result_ty = qbe::Type::from(&expr.ty);
                qfunc.assign_instr(result.clone(), result_ty, instr);
                Ok(Some(result))
            }
        }
    }

    fn generate_expr_assign(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Assign(lhs, rhs) = &expr.kind else {
            unreachable!("ICE: generate_expr_assign called with non-Assign expression")
        };

        let addr = self.address_of(qfunc, lhs)?;
        let value = self.generate_expression(qfunc, rhs)?.ok_or_else(|| {
            CodeGenError::new("Assignment requires a value".to_string(), rhs.span)
        })?;
        let qbe_ty = qbe::Type::from(&rhs.ty);
        qfunc.add_instr(qbe::Instr::Store(qbe_ty, addr, value));

        Ok(None)
    }

    fn generate_expr_return(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Return(ret_expr) = &expr.kind else {
            unreachable!("ICE: generate_expr_return called with non-Return expression")
        };

        let value = match ret_expr {
            Some(expr) => self.generate_expression(qfunc, expr)?,
            None => None,
        };
        qfunc.add_instr(qbe::Instr::Ret(value));

        Ok(None)
    }

    fn generate_expr_control(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        match expr.kind {
            ExprKind::Break => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("break outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.break_label.clone()));
                Ok(None)
            }
            ExprKind::Continue => {
                let loop_ctx = self.current_loop().ok_or_else(|| {
                    CodeGenError::new("continue outside of loop".to_string(), expr.span)
                })?;
                qfunc.add_instr(qbe::Instr::Jmp(loop_ctx.continue_label.clone()));
                Ok(None)
            }

            _ => unreachable!(
                "ICE: generate_expr_control called with non-control-flow expression (expected Break or Continue)"
            ),
        }
    }

    fn generate_expr_array(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Array(elements) = &expr.kind else {
            unreachable!("ICE: generate_expr_array called with non-Array expression")
        };

        // Get element type from the array type
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
        let base = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            base.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        for (i, elem) in elements.iter().enumerate() {
            let elem_val = self.generate_expression(qfunc, elem)?.ok_or_else(|| {
                CodeGenError::new("Array element must produce a value".to_string(), elem.span)
            })?;

            let offset = i as u64 * elem_size;
            let addr = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                addr.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(base.clone(), qbe::Value::Const(offset)),
            );

            let elem_qbe_ty = qbe::Type::from(elem_ty);
            qfunc.add_instr(qbe::Instr::Store(elem_qbe_ty, addr, elem_val));
        }

        Ok(Some(base))
    }

    fn generate_expr_repeat(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Repeat(elem, count) = &expr.kind else {
            unreachable!("ICE: generate_expr_repeat called with non-Repeat expression")
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

        // Get element type from the array type
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
        let base = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            base.clone(),
            qbe::Type::Long,
            qbe::Instr::Alloc8(array_size),
        );

        let elem_val = self.generate_expression(qfunc, elem)?.ok_or_else(|| {
            CodeGenError::new("Array element must produce a value".to_string(), elem.span)
        })?;

        for i in 0..count_num {
            let offset = i as u64 * elem_size;
            let addr = qbe::Value::Temporary(self.new_temp());
            qfunc.assign_instr(
                addr.clone(),
                qbe::Type::Long,
                qbe::Instr::Add(base.clone(), qbe::Value::Const(offset)),
            );

            let elem_qbe_ty = qbe::Type::from(elem_ty);
            qfunc.add_instr(qbe::Instr::Store(elem_qbe_ty, addr, elem_val.clone()));
        }

        Ok(Some(base))
    }

    fn generate_expr_index(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Index(..) = &expr.kind else {
            unreachable!("ICE: generate_expr_index called with non-Index expression")
        };

        let addr = self.address_of(qfunc, expr)?;
        let result = qbe::Value::Temporary(self.new_temp());
        let qbe_ty = qbe::Type::from(&expr.ty);
        qfunc.assign_instr(
            result.clone(),
            qbe_ty.clone(),
            qbe::Instr::Load(qbe_ty, addr),
        );

        Ok(Some(result))
    }

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let result = match &expr.kind {
            ExprKind::Literal(..) => Ok(Some(self.generate_expr_literal(expr))),
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
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Call(call) = &expr.kind else {
            unreachable!("ICE: generate_expr_call called with non-Call expression")
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
            let arg_val = self.generate_expression(qfunc, arg)?.ok_or_else(|| {
                CodeGenError::new(
                    "Function argument must produce a value".to_string(),
                    arg.span,
                )
            })?;
            let arg_ty = qbe::Type::from(&arg.ty);
            qbe_args.push((arg_ty, arg_val));
        }

        // If the function returns Unit, don't capture the return value
        if call.callee.ty == Type::Unit {
            qfunc.add_instr(qbe::Instr::Call(func_name, qbe_args, None));
            Ok(None)
        } else {
            let result = qbe::Value::Temporary(self.new_temp());
            let return_ty = qbe::Type::from(&call.callee.ty);
            qfunc.assign_instr(
                result.clone(),
                return_ty,
                qbe::Instr::Call(func_name, qbe_args, None),
            );
            Ok(Some(result))
        }
    }

    fn generate_expr_if(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::If(if_expr) = &expr.kind else {
            unreachable!("ICE: generate_if called with non-If expression")
        };

        let label_id = self.new_label();
        let cond_label = format!("if.{label_id}.cond");
        let then_label = format!("if.{label_id}.then");
        let end_label = format!("if.{label_id}.end");

        // Condition block
        qfunc.add_block(cond_label);
        let cond = self
            .generate_expression(qfunc, &if_expr.cond)?
            .ok_or_else(|| {
                CodeGenError::new(
                    "Condition must produce a value".to_string(),
                    if_expr.cond.span,
                )
            })?;

        match &if_expr.else_body {
            None => {
                // if without else: no return value
                qfunc.add_instr(qbe::Instr::Jnz(cond, then_label.clone(), end_label.clone()));

                qfunc.add_block(then_label);
                self.generate_block(qfunc, &if_expr.then_body)?;
                if if_expr.then_body.ty != Type::Never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                qfunc.add_block(end_label);
                Ok(None)
            }
            Some(else_expr) => {
                // if-else: can return value
                let else_label = format!("if.{label_id}.else");

                qfunc.add_instr(qbe::Instr::Jnz(
                    cond,
                    then_label.clone(),
                    else_label.clone(),
                ));

                // if the result type is `Never`, no `Instr::Jmp` or `Instr::Phi` need to be
                // generated
                if expr.ty == Type::Never {
                    qfunc.add_block(then_label.clone());
                    self.generate_block(qfunc, &if_expr.then_body)?;

                    qfunc.add_block(else_label.clone());
                    self.generate_expression(qfunc, else_expr)?;

                    qfunc.add_block(end_label);
                    return Ok(None);
                }

                let qbe_ty = qbe::Type::from(&expr.ty);

                // Then block
                qfunc.add_block(then_label.clone());
                let then_result = self.generate_block(qfunc, &if_expr.then_body)?.map(|val| {
                    let temp = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(temp.clone(), qbe_ty.clone(), qbe::Instr::Copy(val));
                    temp
                });

                // Record the actual predecessor block for then branch and add jump if needed
                let last_block = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty after generating then block");
                let then_predecessor = last_block.label.clone();
                if if_expr.then_body.ty != Type::Never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // Else expression (can be a block or another if)
                qfunc.add_block(else_label.clone());
                let else_result = self.generate_expression(qfunc, else_expr)?.map(|val| {
                    let temp = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(temp.clone(), qbe_ty.clone(), qbe::Instr::Copy(val));
                    temp
                });

                // Record the actual predecessor block for else branch and add jump if needed
                let last_block = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty after generating else block");
                let else_predecessor = last_block.label.clone();
                if else_expr.ty != Type::Never {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // End block
                qfunc.add_block(end_label);

                // If a branch is `Never`, use the value from another branch.
                // If both branches have values, use phi.
                match (
                    then_result,
                    else_result,
                    &if_expr.then_body.ty,
                    &else_expr.ty,
                ) {
                    (Some(then_val), Some(else_val), _, _) => {
                        let result = qbe::Value::Temporary(self.new_temp());
                        qfunc.assign_instr(
                            result.clone(),
                            qbe_ty,
                            qbe::Instr::Phi(then_predecessor, then_val, else_predecessor, else_val),
                        );
                        Ok(Some(result))
                    }
                    (Some(then_val), None, _, Type::Never) => Ok(Some(then_val)),
                    (None, Some(else_val), Type::Never, _) => Ok(Some(else_val)),
                    _ => Ok(None),
                }
            }
        }
    }

    fn generate_expr_while(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::While(while_expr) = &expr.kind else {
            unreachable!("ICE: generate_expr_while called with non-While expression")
        };

        let label_id = self.new_label();
        let cond_label = format!("while.{label_id}.cond");
        let body_label = format!("while.{label_id}.body");
        let end_label = format!("while.{label_id}.end");

        // Condition block
        qfunc.add_block(cond_label.clone());
        let cond_val = self
            .generate_expression(qfunc, &while_expr.cond)?
            .ok_or_else(|| {
                CodeGenError::new(
                    "Condition must produce a value".to_string(),
                    while_expr.cond.span,
                )
            })?;
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

        Ok(None)
    }

    fn generate_expr_land(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Binary(BinaryOp::And, left, right) = &expr.kind else {
            unreachable!("ICE: generate_logical_and called with non-And expression")
        };

        let result_ty = qbe::Type::from(&expr.ty);

        let left_val = self.generate_expression(qfunc, left)?.ok_or_else(|| {
            CodeGenError::new("Left operand must produce a value".to_string(), left.span)
        })?;

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
        let right_val = self.generate_expression(qfunc, right)?.ok_or_else(|| {
            CodeGenError::new("Right operand must produce a value".to_string(), right.span)
        })?;
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
        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            result.clone(),
            result_ty,
            qbe::Instr::Phi(rhs_label, right_temp, false_label, false_temp),
        );

        Ok(Some(result))
    }

    fn generate_expr_lor(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr<Type>,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let ExprKind::Binary(BinaryOp::Or, left, right) = &expr.kind else {
            unreachable!("ICE: generate_logical_or called with non-Or expression")
        };

        let result_ty = qbe::Type::from(&expr.ty);

        let left_val = self.generate_expression(qfunc, left)?.ok_or_else(|| {
            CodeGenError::new("Left operand must produce a value".to_string(), left.span)
        })?;

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
        let right_val = self.generate_expression(qfunc, right)?.ok_or_else(|| {
            CodeGenError::new("Right operand must produce a value".to_string(), right.span)
        })?;
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
        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            result.clone(),
            result_ty,
            qbe::Instr::Phi(rhs_label, right_temp, true_label, true_temp),
        );

        Ok(Some(result))
    }
}
