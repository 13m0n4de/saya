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
    scopes: Vec<HashMap<String, VarKind>>,
    loops: Vec<LoopContext>,
    constants: HashMap<String, i64>,
    data_defs: Vec<qbe::DataDef<'static>>,
    globals: HashSet<String>,
}

impl CodeGen {
    pub fn new() -> Self {
        Self {
            temp_counter: 0,
            label_counter: 0,
            scopes: Vec::new(),
            loops: Vec::new(),
            constants: HashMap::new(),
            data_defs: Vec::new(),
            globals: HashSet::new(),
        }
    }

    pub fn generate(&mut self, prog: &Program) -> Result<String, CodeGenError> {
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
            if let Item::Function(func) = item {
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

    fn type_to_qbe(ty: &Ty) -> qbe::Type<'static> {
        match ty {
            Ty::I64 => qbe::Type::Long,
            Ty::Array(_, _) => qbe::Type::Long,
        }
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

    fn type_size(ty: &Ty) -> usize {
        match ty {
            Ty::I64 => 8,
            Ty::Array(elem_ty, len) => Self::type_size(elem_ty) * len,
        }
    }

    fn eval_const_expr(&self, expr: &Expr) -> Result<i64, CodeGenError> {
        match &expr.kind {
            ExprKind::Literal(n) => Ok(*n),
            ExprKind::Ident(name) => self.constants.get(name).copied().ok_or_else(|| {
                CodeGenError::new(
                    format!("Constant `{name}` not found or is not a constant expression"),
                    expr.span,
                )
            }),
            ExprKind::Unary(op, operand) => match op {
                UnaryOp::Neg => Ok(-self.eval_const_expr(operand)?),
            },
            ExprKind::Binary(op, left, right) => {
                let l = self.eval_const_expr(left)?;
                let r = self.eval_const_expr(right)?;

                Ok(match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => {
                        if r == 0 {
                            return Err(CodeGenError::new(
                                "Division by zero in constant expression".to_string(),
                                expr.span,
                            ));
                        }
                        l / r
                    }
                    BinaryOp::Rem => {
                        if r == 0 {
                            return Err(CodeGenError::new(
                                "Division by zero in constant expression".to_string(),
                                expr.span,
                            ));
                        }
                        l % r
                    }

                    BinaryOp::BitAnd => l & r,
                    BinaryOp::BitOr => l | r,

                    BinaryOp::Lt => i64::from(l < r),
                    BinaryOp::Le => i64::from(l <= r),
                    BinaryOp::Gt => i64::from(l > r),
                    BinaryOp::Ge => i64::from(l >= r),
                    BinaryOp::Eq => i64::from(l == r),
                    BinaryOp::Ne => i64::from(l != r),

                    BinaryOp::And => i64::from(l != 0 && r != 0),
                    BinaryOp::Or => i64::from(l != 0 || r != 0),
                })
            }
            _ => Err(CodeGenError::new(
                "Not a constant expression".to_string(),
                expr.span,
            )),
        }
    }

    fn generate_static(&mut self, static_def: &StaticDef) -> Result<(), CodeGenError> {
        let value = self.eval_const_expr(&static_def.init)?;

        self.data_defs.push(qbe::DataDef::new(
            qbe::Linkage::private(),
            static_def.name.clone(),
            None,
            vec![(qbe::Type::Long, qbe::DataItem::Const(value.cast_unsigned()))],
        ));

        Ok(())
    }

    fn generate_function(
        &mut self,
        func: &FunctionDef,
    ) -> Result<qbe::Function<'static>, CodeGenError> {
        self.push_scope();

        for param in &func.params {
            self.insert_param(param.name.clone());
        }

        let params = func
            .params
            .iter()
            .map(|param| {
                let ty = Self::type_to_qbe(&param.type_name);
                let value = qbe::Value::Temporary(param.name.clone());
                (ty, value)
            })
            .collect();

        let return_ty = Some(Self::type_to_qbe(&func.return_type));

        let mut qfunc =
            qbe::Function::new(qbe::Linkage::public(), func.name.clone(), params, return_ty);

        qfunc.add_block("start");
        let block_value = self.generate_block(&mut qfunc, &func.body)?;

        // If block produces a value, return it
        // Otherwise, the block ends with return/break/continue
        if block_value.is_some() {
            qfunc.add_instr(qbe::Instr::Ret(block_value));
        }

        self.pop_scope();

        Ok(qfunc)
    }

    fn generate_block(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        block: &Block,
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
        let_stmt: &Let,
    ) -> Result<(), CodeGenError> {
        let qbe_ty = Self::type_to_qbe(&let_stmt.ty);

        let addr = qbe::Value::Temporary(let_stmt.name.clone());
        qfunc.assign_instr(addr.clone(), qbe::Type::Long, qbe::Instr::Alloc8(8));

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

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        match &expr.kind {
            ExprKind::Literal(lit) => Ok(Some(qbe::Value::Const((*lit).cast_unsigned()))),
            ExprKind::Ident(name) => {
                if let Some(value) = self.constants.get(name) {
                    return Ok(Some(qbe::Value::Const(value.cast_unsigned())));
                }

                if let Some(var_kind) = self.lookup_var(name) {
                    match var_kind {
                        VarKind::Param => {
                            return Ok(Some(qbe::Value::Temporary(name.clone())));
                        }
                        VarKind::Local => {
                            let addr = qbe::Value::Temporary(name.clone());
                            let result = qbe::Value::Temporary(self.new_temp());
                            qfunc.assign_instr(
                                result.clone(),
                                qbe::Type::Long,
                                qbe::Instr::Load(qbe::Type::Long, addr),
                            );
                            return Ok(Some(result));
                        }
                    }
                }

                if self.globals.contains(name) {
                    let addr = qbe::Value::Global(name.clone());
                    let result = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(
                        result.clone(),
                        qbe::Type::Long,
                        qbe::Instr::Load(qbe::Type::Long, addr),
                    );
                    return Ok(Some(result));
                }

                Err(CodeGenError::new(
                    format!("Cannot find value `{name}` in this scope"),
                    expr.span,
                ))
            }
            ExprKind::Call(call) => self.generate_call(qfunc, call),
            ExprKind::Unary(unop, expr) => match unop {
                UnaryOp::Neg => {
                    let operand = self.generate_expression(qfunc, expr)?.ok_or_else(|| {
                        CodeGenError::new("Expression must produce a value".to_string(), expr.span)
                    })?;
                    let result = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(result.clone(), qbe::Type::Long, qbe::Instr::Neg(operand));
                    Ok(Some(result))
                }
            },
            ExprKind::Binary(binop, expr1, expr2) => match binop {
                BinaryOp::And => self.generate_logical_and(qfunc, expr1, expr2),
                BinaryOp::Or => self.generate_logical_or(qfunc, expr1, expr2),
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

                        cmp => qbe::Instr::Cmp(
                            qbe::Type::Long,
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
                        ),
                    };

                    qfunc.assign_instr(result.clone(), qbe::Type::Long, instr);
                    Ok(Some(result))
                }
            },
            ExprKind::Assign(name, assign_expr) => {
                let rhs_val = self
                    .generate_expression(qfunc, assign_expr)?
                    .ok_or_else(|| {
                        CodeGenError::new(
                            "Assignment requires a value".to_string(),
                            assign_expr.span,
                        )
                    })?;

                let addr = if self.globals.contains(name) {
                    qbe::Value::Global(name.clone())
                } else {
                    qbe::Value::Temporary(name.clone())
                };
                qfunc.add_instr(qbe::Instr::Store(qbe::Type::Long, addr, rhs_val));

                Ok(None)
            }
            ExprKind::Return(ret_expr) => {
                let value = match ret_expr {
                    Some(expr) => self.generate_expression(qfunc, expr)?,
                    None => None,
                };
                qfunc.add_instr(qbe::Instr::Ret(value));
                Ok(None)
            }
            ExprKind::Block(block) => self.generate_block(qfunc, block),
            ExprKind::If(if_expr) => self.generate_if(qfunc, if_expr),
            ExprKind::While(while_expr) => self.generate_while(qfunc, while_expr),
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
            ExprKind::Array(elements) => {
                let elem_size = Self::type_size(&Ty::I64);
                let array_size = elements.len() * elem_size;
                let base = qbe::Value::Temporary(self.new_temp());
                qfunc.assign_instr(
                    base.clone(),
                    qbe::Type::Long,
                    qbe::Instr::Alloc8(array_size as u64),
                );

                for (i, elem) in elements.iter().enumerate() {
                    let elem_val = self.generate_expression(qfunc, elem)?.ok_or_else(|| {
                        CodeGenError::new(
                            "Array element must produce a value".to_string(),
                            elem.span,
                        )
                    })?;

                    let offset = (i * Self::type_size(&Ty::I64)) as u64;
                    let addr = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(
                        addr.clone(),
                        qbe::Type::Long,
                        qbe::Instr::Add(base.clone(), qbe::Value::Const(offset)),
                    );

                    qfunc.add_instr(qbe::Instr::Store(qbe::Type::Long, addr, elem_val));
                }

                Ok(Some(base))
            }
        }
    }

    fn generate_call(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        call: &Call,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        let func_name = match &call.callee.kind {
            ExprKind::Ident(name) => name.clone(),
            _ => {
                return Err(CodeGenError::new(
                    "Complex callee expressions not yet supported".to_string(),
                    call.callee.span,
                ));
            }
        };

        let mut arg_values = Vec::new();
        for arg in &call.args {
            let arg_val = self.generate_expression(qfunc, arg)?.ok_or_else(|| {
                CodeGenError::new(
                    "Function argument must produce a value".to_string(),
                    arg.span,
                )
            })?;
            arg_values.push(arg_val);
        }

        let qbe_args: Vec<(qbe::Type, qbe::Value)> = arg_values
            .into_iter()
            .map(|val| (qbe::Type::Long, val))
            .collect();

        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            result.clone(),
            qbe::Type::Long,
            qbe::Instr::Call(func_name, qbe_args, None),
        );

        Ok(Some(result))
    }

    fn generate_if(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        if_expr: &If,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
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

        match &if_expr.else_block {
            None => {
                // if without else: no return value
                qfunc.add_instr(qbe::Instr::Jnz(cond, then_label.clone(), end_label.clone()));

                qfunc.add_block(then_label);
                self.generate_block(qfunc, &if_expr.then_block)?;

                if !qfunc.blocks.last().is_some_and(qbe::Block::jumps) {
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

                // Then block
                qfunc.add_block(then_label.clone());
                let then_result = self.generate_block(qfunc, &if_expr.then_block)?.map(|val| {
                    let temp = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(temp.clone(), qbe::Type::Long, qbe::Instr::Copy(val));
                    temp
                });

                // Record the actual predecessor block for then branch and add jump if needed
                let last_block = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty after generating then block");
                let then_predecessor = last_block.label.clone();
                if !last_block.jumps() {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // Else expression (can be a block or another if)
                qfunc.add_block(else_label.clone());
                let else_result = self.generate_expression(qfunc, else_expr)?.map(|val| {
                    let temp = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(temp.clone(), qbe::Type::Long, qbe::Instr::Copy(val));
                    temp
                });

                // Record the actual predecessor block for else branch and add jump if needed
                let last_block = qfunc
                    .blocks
                    .last()
                    .expect("ICE: blocks should not be empty after generating else block");
                let else_predecessor = last_block.label.clone();
                if !last_block.jumps() {
                    qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));
                }

                // End block
                qfunc.add_block(end_label);

                // If both branches return a value, merge them with phi
                match (then_result, else_result) {
                    (Some(then_val), Some(else_val)) => {
                        let result = qbe::Value::Temporary(self.new_temp());
                        qfunc.assign_instr(
                            result.clone(),
                            qbe::Type::Long,
                            qbe::Instr::Phi(then_predecessor, then_val, else_predecessor, else_val),
                        );
                        Ok(Some(result))
                    }
                    _ => Ok(None),
                }
            }
        }
    }

    fn generate_while(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        while_expr: &While,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
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

    fn generate_logical_and(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        left: &Expr,
        right: &Expr,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
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
            qbe::Type::Long,
            qbe::Instr::Copy(right_val),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(false_label.clone());
        let false_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            false_temp.clone(),
            qbe::Type::Long,
            qbe::Instr::Copy(qbe::Value::Const(0)),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(end_label);
        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            result.clone(),
            qbe::Type::Long,
            qbe::Instr::Phi(rhs_label, right_temp, false_label, false_temp),
        );

        Ok(Some(result))
    }

    fn generate_logical_or(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        left: &Expr,
        right: &Expr,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
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
            qbe::Type::Long,
            qbe::Instr::Copy(right_val),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(true_label.clone());
        let true_temp = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            true_temp.clone(),
            qbe::Type::Long,
            qbe::Instr::Copy(qbe::Value::Const(1)),
        );
        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

        qfunc.add_block(end_label);
        let result = qbe::Value::Temporary(self.new_temp());
        qfunc.assign_instr(
            result.clone(),
            qbe::Type::Long,
            qbe::Instr::Phi(rhs_label, right_temp, true_label, true_temp),
        );

        Ok(Some(result))
    }
}
