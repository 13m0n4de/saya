use std::{fs::read_to_string, str::Chars};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Fn,
    Return,

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
pub enum Stmt {
    Expr(Expr),
    Semi(Expr),
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
    Return(Option<Box<Expr>>),
    Block(Block),
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

            let expr = self.parse_expression()?;

            if self.eat(TokenKind::Semi)? {
                stmts.push(Stmt::Semi(expr));
            } else {
                stmts.push(Stmt::Expr(expr));
                break;
            }
        }

        self.expect(TokenKind::CloseBrace)?;

        Ok(Block {
            stmts,
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

        self.parse_bitwise_or()
    }

    fn parse_bitwise_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitwise_and()?;

        while self.current.kind == TokenKind::Or {
            self.advance()?;
            let right = self.parse_bitwise_and()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(BinaryOp::BitOr, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_equality()?;

        while self.current.kind == TokenKind::And {
            self.advance()?;
            let right = self.parse_equality()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(BinaryOp::BitAnd, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_relational()?;

        loop {
            let op = match self.current.kind {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::Ne => BinaryOp::Ne,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_relational()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Le => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::Ge => BinaryOp::Ge,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_additive()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_multiplicative()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.current.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Rem,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_unary()?;

            let span = left.span;
            left = Expr {
                kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let start_span = self.current.span;

        match &self.current.kind {
            TokenKind::Minus => {
                self.advance()?;
                let expr = Box::new(self.parse_unary()?);
                Ok(Expr {
                    kind: ExprKind::Unary(UnaryOp::Neg, expr),
                    span: start_span,
                })
            }
            _ => self.parse_primary(),
        }
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

pub struct CodeGen {
    temp_counter: usize,
}

impl CodeGen {
    fn new() -> Self {
        Self { temp_counter: 0 }
    }

    fn new_temp(&mut self) -> String {
        let name = format!("temp.{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn type_to_qbe(ty: &Ty) -> qbe::Type<'static> {
        match ty {
            Ty::I64 => qbe::Type::Long,
        }
    }

    fn generate(&mut self, prog: &Program) -> String {
        let mut module = qbe::Module::new();

        for func in &prog.functions {
            module.add_function(self.generate_function(func));
        }

        module.to_string()
    }

    fn generate_function(&mut self, func: &FunctionDef) -> qbe::Function<'static> {
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
        self.generate_block(&mut qfunc, &func.body);

        qfunc
    }

    fn generate_block(&mut self, qfunc: &mut qbe::Function<'static>, block: &Block) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr) | Stmt::Semi(expr) => {
                    self.generate_expression(qfunc, expr);
                }
            }
        }
    }

    fn generate_expression(
        &mut self,
        qfunc: &mut qbe::Function<'static>,
        expr: &Expr,
    ) -> Option<qbe::Value> {
        match &expr.kind {
            ExprKind::Literal(lit) => Some(qbe::Value::Const(*lit as u64)),
            ExprKind::Ident(name) => Some(qbe::Value::Temporary(name.clone())),
            ExprKind::Unary(unop, expr) => match unop {
                UnaryOp::Neg => {
                    let operand = self.generate_expression(qfunc, expr)?;
                    let result = qbe::Value::Temporary(self.new_temp());
                    qfunc.assign_instr(result.clone(), qbe::Type::Long, qbe::Instr::Neg(operand));
                    Some(result)
                }
            },
            ExprKind::Binary(binop, expr1, expr2) => {
                let operand1 = self.generate_expression(qfunc, expr1)?;
                let operand2 = self.generate_expression(qfunc, expr2)?;
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
                Some(result)
            }
            ExprKind::Return(ret_expr) => {
                let value = match ret_expr {
                    Some(expr) => self.generate_expression(qfunc, expr),
                    None => None,
                };
                qfunc.add_instr(qbe::Instr::Ret(value));
                None
            }
            ExprKind::Block(block) => {
                let mut result = None;
                for stmt in &block.stmts {
                    match stmt {
                        Stmt::Semi(expr) => {
                            self.generate_expression(qfunc, expr);
                        }
                        Stmt::Expr(expr) => {
                            result = self.generate_expression(qfunc, expr);
                        }
                    }
                }
                result
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
            let qbe_il = code_gen.generate(&program);
            println!("{qbe_il}");
            std::fs::write("out.ssa", qbe_il).unwrap();
        }
        Err(e) => {
            eprintln!(
                "Parse error at {}:{}: {}",
                e.span.line, e.span.column, e.message
            );
        }
    }
}
