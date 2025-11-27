use std::{collections::HashMap, fs::read_to_string, str::Chars};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Fn,     // fn
    Return, // return
    Let,    // let
    If,     // if
    Else,   // else

    Ident(String),
    Integer(i64),

    Plus,    // +
    Minus,   // -
    Star,    // *
    Slash,   // /
    Percent, // %

    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=

    And, // &
    Or,  // |
    Eq,  // =
    Not, // !

    EqEq, // ==
    Ne,   // !=

    AndAnd, // &&
    OrOr,   // ||

    OpenParen,  // (
    CloseParen, // )
    OpenBrace,  // {
    CloseBrace, // }
    Comma,      // ,
    Colon,      // :
    Semi,       // ;
    Arrow,      // ->

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub line: usize,
    pub column: usize,
}

#[derive(Debug)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl LexError {
    pub fn new(message: String, span: Span) -> Self {
        Self { message, span }
    }
}

pub struct Lexer<'a> {
    chars: Chars<'a>,
    current: Option<char>,
    peek: Option<char>,
    span: Span,
    start_span: Span,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let current = chars.next();
        let peek = chars.next();

        let span = Span { line: 1, column: 1 };

        Self {
            chars,
            current,
            peek,
            span,
            start_span: span,
        }
    }

    fn advance(&mut self) {
        if let Some(ch) = self.current {
            if ch == '\n' {
                self.span.line += 1;
                self.span.column = 1;
            } else {
                self.span.column += 1;
            }
        }

        self.current = self.peek;
        self.peek = self.chars.next();
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace();

        self.start_span = self.span;

        let kind = match self.current {
            Some(ch) if ch.is_ascii_alphabetic() => self.read_identifier(),
            Some(ch) if ch.is_ascii_digit() => self.read_number()?,

            Some('(') => {
                self.advance();
                TokenKind::OpenParen
            }
            Some(')') => {
                self.advance();
                TokenKind::CloseParen
            }
            Some('{') => {
                self.advance();
                TokenKind::OpenBrace
            }
            Some('}') => {
                self.advance();
                TokenKind::CloseBrace
            }
            Some(',') => {
                self.advance();
                TokenKind::Comma
            }
            Some(':') => {
                self.advance();
                TokenKind::Colon
            }
            Some(';') => {
                self.advance();
                TokenKind::Semi
            }

            Some('+') => {
                self.advance();
                TokenKind::Plus
            }
            Some('-') => {
                self.advance();
                if self.current == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            Some('*') => {
                self.advance();
                TokenKind::Star
            }
            Some('/') => {
                self.advance();
                TokenKind::Slash
            }
            Some('%') => {
                self.advance();
                TokenKind::Percent
            }

            Some('<') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                }
            }
            Some('>') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }

            Some('=') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    TokenKind::EqEq
                } else {
                    TokenKind::Eq
                }
            }
            Some('!') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    TokenKind::Ne
                } else {
                    TokenKind::Not
                }
            }

            Some('&') => {
                self.advance();
                if self.current == Some('&') {
                    self.advance();
                    TokenKind::AndAnd
                } else {
                    TokenKind::And
                }
            }
            Some('|') => {
                self.advance();
                if self.current == Some('|') {
                    self.advance();
                    TokenKind::OrOr
                } else {
                    TokenKind::Or
                }
            }

            Some(ch) => {
                return Err(LexError::new(
                    format!("Unexpected character: '{ch}'"),
                    self.start_span,
                ));
            }

            None => TokenKind::Eof,
        };

        Ok(Token {
            kind,
            span: self.start_span,
        })
    }

    fn read_number(&mut self) -> Result<TokenKind, LexError> {
        let mut num = String::new();

        while let Some(ch) = self.current {
            if ch.is_ascii_digit() {
                num.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match num.parse::<i64>() {
            Ok(value) => Ok(TokenKind::Integer(value)),
            Err(_) => Err(LexError::new(
                format!("Integer literal is too large: {num}"),
                self.start_span,
            )),
        }
    }

    fn read_identifier(&mut self) -> TokenKind {
        let mut ident = String::new();

        while let Some(ch) = self.current {
            if ch.is_ascii_alphanumeric() {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match ident.as_str() {
            "fn" => TokenKind::Fn,
            "return" => TokenKind::Return,
            "let" => TokenKind::Let,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            _ => TokenKind::Ident(ident),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub functions: Vec<FunctionDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Ty,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_name: Ty,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    I64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    kind: StmtKind,
    span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Expr(Expr),
    Semi(Expr),
    Let(Let),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Let {
    pub name: String,
    pub ty: Ty,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(i64),
    Ident(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Assign(String, Box<Expr>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(If),
}

#[derive(Debug, Clone, PartialEq)]
pub struct If {
    pub cond: Box<Expr>,
    pub then_block: Box<Block>,
    pub else_block: Option<Box<Block>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Rem,    // %
    Lt,     // <
    Le,     // <=
    Gt,     // >
    Ge,     // >=
    Eq,     // ==
    Ne,     // !=
    BitAnd, // &
    BitOr,  // |
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: String, span: Span) -> Self {
        Self { message, span }
    }
}

impl From<LexError> for ParseError {
    fn from(lex_error: LexError) -> Self {
        ParseError {
            message: format!("Lexer error: {}", lex_error.message),
            span: lex_error.span,
        }
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    pub fn new(mut lexer: Lexer<'a>) -> Result<Self, ParseError> {
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: TokenKind) -> Result<(), ParseError> {
        if self.current.kind != expected {
            return Err(ParseError::new(
                format!("Expected {:?}, found {:?}", expected, self.current.kind),
                self.current.span,
            ));
        }
        self.advance()?;
        Ok(())
    }

    fn expect_identifier(&mut self) -> Result<String, ParseError> {
        if let TokenKind::Ident(name) = &self.current.kind {
            let name = name.clone();
            self.advance()?;
            Ok(name)
        } else {
            Err(ParseError::new(
                format!("Expected identifier, found {:?}", self.current.kind),
                self.current.span,
            ))
        }
    }

    fn parse_type(&mut self) -> Result<Ty, ParseError> {
        let type_span = self.current.span;
        let name = self.expect_identifier()?;
        match name.as_str() {
            "i64" => Ok(Ty::I64),
            _ => Err(ParseError::new(format!("Unknown type: {name}"), type_span)),
        }
    }

    fn eat(&mut self, token: TokenKind) -> Result<bool, ParseError> {
        if self.current.kind == token {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn parse(&mut self) -> Result<Program, ParseError> {
        self.parse_program()
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut functions = Vec::new();

        loop {
            match self.current.kind {
                TokenKind::Fn => functions.push(self.parse_function()?),
                TokenKind::Eof => break,
                _ => {
                    return Err(ParseError::new(
                        format!("Unexpected token: {:?}", self.current.kind),
                        self.current.span,
                    ));
                }
            }
        }

        Ok(Program { functions })
    }

    fn parse_function(&mut self) -> Result<FunctionDef, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Fn)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::OpenParen)?;

        let params = if self.current.kind != TokenKind::CloseParen {
            self.parse_param_list()?
        } else {
            Vec::new()
        };

        self.expect(TokenKind::CloseParen)?;

        self.expect(TokenKind::Arrow)?;

        let return_type = self.parse_type()?;

        let body = self.parse_block()?;

        Ok(FunctionDef {
            name,
            params,
            return_type,
            body,
            span: start_span,
        })
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();

        params.push(self.parse_param()?);

        while self.eat(TokenKind::Comma)? {
            params.push(self.parse_param()?);
        }

        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let start_span = self.current.span;

        let name = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let type_name = self.parse_type()?;

        Ok(Param {
            name,
            type_name,
            span: start_span,
        })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::OpenBrace)?;

        let mut stmts = Vec::new();

        loop {
            if self.current.kind == TokenKind::CloseBrace {
                break;
            }

            if self.current.kind == TokenKind::Let {
                let stmt = self.parse_let_statement()?;
                stmts.push(stmt);
                continue;
            }

            let expr = self.parse_expression()?;
            let expr_span = expr.span;

            if self.eat(TokenKind::Semi)? {
                // Explicit semicolon: statement
                stmts.push(Stmt {
                    kind: StmtKind::Semi(expr),
                    span: expr_span,
                });
            } else if self.current.kind == TokenKind::CloseBrace {
                // No semicolon, followed by '}': tail expression
                stmts.push(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: expr_span,
                });
                break;
            } else if matches!(expr.kind, ExprKind::Block(_)) {
                // Block expressions can omit semicolons
                stmts.push(Stmt {
                    kind: StmtKind::Semi(expr),
                    span: expr_span,
                });
            } else {
                // Other expressions require semicolons
                return Err(ParseError::new(
                    format!(
                        "Expected ';' after expression (found {:?})",
                        self.current.kind
                    ),
                    self.current.span,
                ));
            }
        }

        self.expect(TokenKind::CloseBrace)?;

        Ok(Block {
            stmts,
            span: start_span,
        })
    }

    fn parse_let_statement(&mut self) -> Result<Stmt, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Let)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::Colon)?;

        let ty = self.parse_type()?;

        self.expect(TokenKind::Eq)?;

        let init = self.parse_expression()?;

        self.expect(TokenKind::Semi)?;

        Ok(Stmt {
            kind: StmtKind::Let(Let {
                name,
                ty,
                init,
                span: start_span,
            }),
            span: start_span,
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        let start_span = self.current.span;

        if self.current.kind == TokenKind::Return {
            self.advance()?;

            let ret_expr = if self.current.kind == TokenKind::Semi
                || self.current.kind == TokenKind::CloseBrace
            {
                None
            } else {
                Some(Box::new(self.parse_expression()?))
            };

            return Ok(Expr {
                kind: ExprKind::Return(ret_expr),
                span: start_span,
            });
        }

        self.parse_expr_assign()
    }

    fn parse_expr_assign(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_expr_bp(0)?;

        if self.current.kind == TokenKind::Eq {
            let name = match &lhs.kind {
                ExprKind::Ident(n) => n.clone(),
                _ => {
                    return Err(ParseError::new(
                        "Left side of assignment must be an identifier".to_string(),
                        lhs.span,
                    ));
                }
            };

            self.advance()?;
            let rhs = self.parse_expr_assign()?;

            Ok(Expr {
                kind: ExprKind::Assign(name, Box::new(rhs)),
                span: lhs.span,
            })
        } else {
            Ok(lhs)
        }
    }

    fn infix_binding_power(&self, token: &TokenKind) -> Option<(u8, u8)> {
        let bp = match token {
            TokenKind::Or => (10, 11),                   // |
            TokenKind::And => (20, 21),                  // &
            TokenKind::EqEq | TokenKind::Ne => (30, 31), // == !=
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => (40, 41), // < <= > >=
            TokenKind::Plus | TokenKind::Minus => (50, 51), // + -
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (60, 61), // * / %
            _ => return None,
        };
        Some(bp)
    }

    fn prefix_binding_power(&self, token: &TokenKind) -> Option<u8> {
        match token {
            TokenKind::Minus => Some(70), // -
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = if let Some(prefix_bp) = self.prefix_binding_power(&self.current.kind) {
            // Prefix operator
            let start_span = self.current.span;
            let op = match self.current.kind {
                TokenKind::Minus => UnaryOp::Neg,
                _ => unreachable!(),
            };
            self.advance()?;
            let rhs = self.parse_expr_bp(prefix_bp)?;
            Expr {
                kind: ExprKind::Unary(op, Box::new(rhs)),
                span: start_span,
            }
        } else {
            // Primary expression
            self.parse_primary()?
        };

        // Infix operators
        loop {
            let op_token = &self.current.kind.clone();

            let (left_bp, right_bp) = match self.infix_binding_power(op_token) {
                Some(bp) => bp,
                None => break,
            };

            if left_bp < min_bp {
                break;
            }

            let op = match op_token {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Rem,
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Le => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::Ge => BinaryOp::Ge,
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::Ne => BinaryOp::Ne,
                TokenKind::And => BinaryOp::BitAnd,
                TokenKind::Or => BinaryOp::BitOr,
                _ => unreachable!(),
            };

            let span = lhs.span;
            self.advance()?;
            let rhs = self.parse_expr_bp(right_bp)?;

            lhs = Expr {
                kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }

        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let start_span = self.current.span;

        let kind = match &self.current.kind {
            TokenKind::Integer(val) => {
                let val = *val;
                self.advance()?;
                ExprKind::Literal(val)
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance()?;
                ExprKind::Ident(name)
            }
            TokenKind::OpenParen => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::CloseParen)?;
                return Ok(expr);
            }
            TokenKind::OpenBrace => {
                let block = self.parse_block()?;
                ExprKind::Block(block)
            }
            TokenKind::If => {
                let if_span = self.current.span;
                self.advance()?;

                let cond = Box::new(self.parse_expression()?);
                let then_block = Box::new(self.parse_block()?);
                let else_block = if self.eat(TokenKind::Else)? {
                    Some(Box::new(self.parse_block()?))
                } else {
                    None
                };

                ExprKind::If(If {
                    cond,
                    then_block,
                    else_block,
                    span: if_span,
                })
            }
            _ => {
                return Err(ParseError::new(
                    format!("Expected expression, found {:?}", self.current.kind),
                    self.current.span,
                ));
            }
        };

        Ok(Expr {
            kind,
            span: start_span,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarKind {
    Param,
    Local,
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

pub struct CodeGen {
    temp_counter: usize,
    label_counter: usize,
    scopes: Vec<HashMap<String, VarKind>>,
}

impl CodeGen {
    fn new() -> Self {
        Self {
            temp_counter: 0,
            label_counter: 0,
            scopes: Vec::new(),
        }
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

    fn type_to_qbe(ty: &Ty) -> qbe::Type<'static> {
        match ty {
            Ty::I64 => qbe::Type::Long,
        }
    }

    fn generate(&mut self, prog: &Program) -> Result<String, CodeGenError> {
        let mut module = qbe::Module::new();

        for func in &prog.functions {
            module.add_function(self.generate_function(func)?);
        }

        Ok(module.to_string())
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
        self.generate_block(&mut qfunc, &func.body)?;

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

        if let Some(init_val) = self.generate_expression(qfunc, &let_stmt.init)? {
            qfunc.add_instr(qbe::Instr::Store(qbe_ty, addr, init_val));
        }
        Ok(())
    }

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Result<Option<qbe::Value>, CodeGenError> {
        match &expr.kind {
            ExprKind::Literal(lit) => Ok(Some(qbe::Value::Const(*lit as u64))),
            ExprKind::Ident(name) => match self.lookup_var(name) {
                Some(VarKind::Param) => Ok(Some(qbe::Value::Temporary(name.clone()))),
                Some(VarKind::Local) => {
                    let addr = qbe::Value::Temporary(name.clone());
                    let result = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(
                        result.clone(),
                        qbe::Type::Long,
                        qbe::Instr::Load(qbe::Type::Long, addr),
                    );
                    Ok(Some(result))
                }
                None => Err(CodeGenError::new(
                    format!("Cannot find value `{}` in this scope", name),
                    expr.span,
                )),
            },
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
            ExprKind::Binary(binop, expr1, expr2) => {
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
            ExprKind::Assign(name, assign_expr) => {
                if let Some(rhs_val) = self.generate_expression(qfunc, assign_expr)? {
                    let addr = qbe::Value::Temporary(name.clone());
                    qfunc.add_instr(qbe::Instr::Store(qbe::Type::Long, addr, rhs_val));
                }
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
            ExprKind::If(if_expr) => {
                let cond = self
                    .generate_expression(qfunc, &if_expr.cond)?
                    .ok_or_else(|| {
                        CodeGenError::new(
                            "Condition must produce a value".to_string(),
                            if_expr.cond.span,
                        )
                    })?;

                let label_id = self.new_label();
                let then_label = format!("if.{}.then", label_id);
                let end_label = format!("if.{}.end", label_id);

                match &if_expr.else_block {
                    None => {
                        // if without else: no return value
                        qfunc.add_instr(qbe::Instr::Jnz(
                            cond,
                            then_label.clone(),
                            end_label.clone(),
                        ));

                        qfunc.add_block(then_label);
                        self.generate_block(qfunc, &if_expr.then_block)?;

                        qfunc.add_block(end_label);
                        Ok(None)
                    }
                    Some(else_block) => {
                        // if-else: can return value
                        let else_label = format!("if.{}.else", label_id);

                        qfunc.add_instr(qbe::Instr::Jnz(
                            cond,
                            then_label.clone(),
                            else_label.clone(),
                        ));

                        // Then block
                        qfunc.add_block(then_label.clone());
                        let then_result =
                            self.generate_block(qfunc, &if_expr.then_block)?.map(|val| {
                                let temp = qbe::Value::Temporary(self.new_temp());
                                qfunc.assign_instr(
                                    temp.clone(),
                                    qbe::Type::Long,
                                    qbe::Instr::Copy(val),
                                );
                                temp
                            });
                        qfunc.add_instr(qbe::Instr::Jmp(end_label.clone()));

                        // Else block
                        qfunc.add_block(else_label.clone());
                        let else_result = self.generate_block(qfunc, else_block)?.map(|val| {
                            let temp = qbe::Value::Temporary(self.new_temp());
                            qfunc.assign_instr(
                                temp.clone(),
                                qbe::Type::Long,
                                qbe::Instr::Copy(val),
                            );
                            temp
                        });

                        // End block
                        qfunc.add_block(end_label);

                        // If both branches return a value, merge them with phi
                        match (then_result, else_result) {
                            (Some(then_val), Some(else_val)) => {
                                let result = qbe::Value::Temporary(self.new_temp());
                                qfunc.assign_instr(
                                    result.clone(),
                                    qbe::Type::Long,
                                    qbe::Instr::Phi(then_label, then_val, else_label, else_val),
                                );
                                Ok(Some(result))
                            }
                            _ => Ok(None),
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    let code = match read_to_string("examples/hello_world.saya") {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Failed to read file: {e}");
            return;
        }
    };

    let lexer = Lexer::new(&code);
    let mut parser = match Parser::new(lexer) {
        Ok(parser) => parser,
        Err(e) => {
            eprintln!(
                "Parse error at {}:{}: {}",
                e.span.line, e.span.column, e.message
            );
            return;
        }
    };

    match parser.parse() {
        Ok(program) => {
            println!("{program:#?}");
            let mut code_gen = CodeGen::new();
            match code_gen.generate(&program) {
                Ok(qbe_il) => {
                    println!("{qbe_il}");
                    std::fs::write("out.ssa", qbe_il).unwrap();
                }
                Err(e) => {
                    eprintln!(
                        "Code generation error at {}:{}: {}",
                        e.span.line, e.span.column, e.message
                    );
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Parse error at {}:{}: {}",
                e.span.line, e.span.column, e.message
            );
        }
    }
}
