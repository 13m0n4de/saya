use std::{error::Error, fmt};

use crate::{
    ast::*,
    lexer::{LexError, Lexer, Token, TokenKind},
    span::Span,
};

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

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "parse error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl Error for ParseError {}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    pub fn new(mut lexer: Lexer<'a>) -> Result<Self, ParseError> {
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    pub fn parse(&mut self) -> Result<Program<()>, ParseError> {
        self.parse_program()
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

    fn eat(&mut self, token: TokenKind) -> Result<bool, ParseError> {
        if self.current.kind == token {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn parse_type_ann(&mut self) -> Result<TypeAnn, ParseError> {
        match self.current.kind.clone() {
            TokenKind::Ident(name) => {
                self.advance()?;
                match name.as_str() {
                    "i64" => Ok(TypeAnn::I64),
                    "str" => Ok(TypeAnn::Str),
                    "bool" => Ok(TypeAnn::Bool),
                    _ => Err(ParseError::new(
                        format!("Unknown type: {name}"),
                        self.current.span,
                    )),
                }
            }
            TokenKind::OpenBracket => {
                self.advance()?;
                let elem_type_ann = Box::new(self.parse_type_ann()?);
                self.expect(TokenKind::Semi)?;

                let size = if let TokenKind::Integer(n) = self.current.kind {
                    if n < 0 {
                        return Err(ParseError::new(
                            "Array size cannot be negative".to_string(),
                            self.current.span,
                        ));
                    }
                    n as usize
                } else {
                    return Err(ParseError::new(
                        "Expected array size".to_string(),
                        self.current.span,
                    ));
                };

                self.advance()?;
                self.expect(TokenKind::CloseBracket)?;
                Ok(TypeAnn::Array(elem_type_ann, size))
            }
            _ => Err(ParseError::new(
                format!("Unknown type: {:?}", self.current.kind),
                self.current.span,
            )),
        }
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();

        loop {
            match self.current.kind {
                TokenKind::Const => items.push(Item::Const(self.parse_const()?)),
                TokenKind::Static => items.push(Item::Static(self.parse_static()?)),
                TokenKind::Fn => items.push(Item::Function(self.parse_function()?)),
                TokenKind::Eof => break,
                _ => {
                    return Err(ParseError::new(
                        format!("Unexpected token: {:?}", self.current.kind),
                        self.current.span,
                    ));
                }
            }
        }

        Ok(Program { items })
    }

    fn parse_const(&mut self) -> Result<ConstDef, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Const)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::Colon)?;

        let type_ann = self.parse_type_ann()?;

        self.expect(TokenKind::Eq)?;

        let init = Box::new(self.parse_expression()?);

        self.expect(TokenKind::Semi)?;

        Ok(ConstDef {
            name,
            type_ann,
            init,
            span: start_span,
        })
    }

    fn parse_static(&mut self) -> Result<StaticDef, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Static)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::Colon)?;

        let type_ann = self.parse_type_ann()?;

        self.expect(TokenKind::Eq)?;

        let init = Box::new(self.parse_expression()?);

        self.expect(TokenKind::Semi)?;

        Ok(StaticDef {
            name,
            type_ann,
            init,
            span: start_span,
        })
    }

    fn parse_function(&mut self) -> Result<FunctionDef, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Fn)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::OpenParen)?;

        let params = if self.current.kind == TokenKind::CloseParen {
            Vec::new()
        } else {
            self.parse_param_list()?
        };

        self.expect(TokenKind::CloseParen)?;

        let return_type_ann = if self.eat(TokenKind::Arrow)? {
            self.parse_type_ann()?
        } else {
            TypeAnn::Unit
        };

        let body = if self.eat(TokenKind::Semi)? {
            None
        } else {
            Some(self.parse_block()?)
        };

        Ok(FunctionDef {
            name,
            params,
            return_type_ann,
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
        let type_ann = self.parse_type_ann()?;

        Ok(Param {
            name,
            type_ann,
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
            } else if matches!(
                expr.kind,
                ExprKind::Block(_) | ExprKind::If(_) | ExprKind::While(_)
            ) {
                // `Block`, `If`, `While` can omit semicolons after `{ }`
                stmts.push(Stmt {
                    kind: StmtKind::Expr(expr),
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
            ty: (),
            stmts,
            span: start_span,
        })
    }

    fn parse_let_statement(&mut self) -> Result<Stmt, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Let)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::Colon)?;

        let type_ann = self.parse_type_ann()?;

        self.expect(TokenKind::Eq)?;

        let init = self.parse_expression()?;

        self.expect(TokenKind::Semi)?;

        Ok(Stmt {
            kind: StmtKind::Let(Let {
                name,
                type_ann,
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
                ty: (),
                kind: ExprKind::Return(ret_expr),
                span: start_span,
            });
        }

        self.parse_expr_assign()
    }

    fn parse_expr_assign(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_expr_bp(0)?;
        let start_span = lhs.span;

        if self.current.kind == TokenKind::Eq {
            self.advance()?;
            let rhs = self.parse_expr_assign()?;

            Ok(Expr {
                ty: (),
                kind: ExprKind::Assign(Box::new(lhs), Box::new(rhs)),
                span: start_span,
            })
        } else {
            Ok(lhs)
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = if let Some(prefix_bp) = self.prefix_binding_power(&self.current.kind) {
            // Prefix operator
            let start_span = self.current.span;
            let op = match self.current.kind {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Bang => UnaryOp::Not,
                _ => unreachable!(),
            };
            self.advance()?;
            let rhs = self.parse_expr_bp(prefix_bp)?;
            Expr {
                ty: (),
                kind: ExprKind::Unary(op, Box::new(rhs)),
                span: start_span,
            }
        } else {
            // Primary expression
            let primary = self.parse_primary()?;

            // Structural expressions (if, while, block) should not participate in infix operators
            // to avoid parsing "if x { } - y" as "(if x { }) minus y"
            if matches!(
                primary.kind,
                ExprKind::If(_) | ExprKind::While(_) | ExprKind::Block(_)
            ) {
                return Ok(primary);
            }

            primary
        };

        loop {
            let op_token = &self.current.kind;

            // Postfix operators
            if let Some(postfix_bp) = self.postfix_binding_power(op_token) {
                if postfix_bp < min_bp {
                    break;
                }

                match op_token {
                    TokenKind::OpenParen => {
                        let span = lhs.span;
                        lhs = Expr {
                            ty: (),
                            kind: ExprKind::Call(self.parse_call(lhs)?),
                            span,
                        }
                    }
                    TokenKind::OpenBracket => {
                        let span = lhs.span;
                        self.advance()?;
                        let index = self.parse_expression()?;
                        self.expect(TokenKind::CloseBracket)?;
                        lhs = Expr {
                            ty: (),
                            kind: ExprKind::Index(Box::new(lhs), Box::new(index)),
                            span,
                        }
                    }
                    _ => unreachable!(),
                }

                continue;
            }

            // Infix operators
            if let Some((left_bp, right_bp)) = self.infix_binding_power(op_token) {
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
                    TokenKind::AndAnd => BinaryOp::And,
                    TokenKind::OrOr => BinaryOp::Or,
                    _ => unreachable!(),
                };

                let span = lhs.span;
                self.advance()?;
                let rhs = self.parse_expr_bp(right_bp)?;

                lhs = Expr {
                    ty: (),
                    kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                    span,
                };

                continue;
            }

            // Neither postfix nor infix, stop parsing
            break;
        }

        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let start_span = self.current.span;

        let kind = match &self.current.kind {
            TokenKind::Integer(val) => {
                let val = *val;
                self.advance()?;
                ExprKind::Literal(Literal::Integer(val))
            }
            TokenKind::String(str) => {
                let val = str.to_owned();
                self.advance()?;
                ExprKind::Literal(Literal::String(val))
            }
            TokenKind::True => {
                self.advance()?;
                ExprKind::Literal(Literal::Bool(true))
            }
            TokenKind::False => {
                self.advance()?;
                ExprKind::Literal(Literal::Bool(false))
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

            TokenKind::If => ExprKind::If(self.parse_if()?),
            TokenKind::While => ExprKind::While(self.parse_while()?),
            TokenKind::Break => {
                self.advance()?;
                ExprKind::Break
            }
            TokenKind::Continue => {
                self.advance()?;
                ExprKind::Continue
            }

            TokenKind::OpenBracket => {
                self.advance()?;

                if self.eat(TokenKind::CloseBracket)? {
                    ExprKind::Array(Vec::new())
                } else {
                    let first_expr = self.parse_expression()?;

                    // [expr; count]
                    if self.eat(TokenKind::Semi)? {
                        let count = self.parse_expression()?;
                        self.expect(TokenKind::CloseBracket)?;
                        ExprKind::Repeat(Box::new(first_expr), Box::new(count))
                    } else {
                        // [expr, expr, ...]
                        let mut elements = vec![first_expr];
                        while self.eat(TokenKind::Comma)? {
                            elements.push(self.parse_expression()?);
                        }
                        self.expect(TokenKind::CloseBracket)?;
                        ExprKind::Array(elements)
                    }
                }
            }

            _ => {
                return Err(ParseError::new(
                    format!("Expected expression, found {:?}", self.current.kind),
                    self.current.span,
                ));
            }
        };

        Ok(Expr {
            ty: (),
            kind,
            span: start_span,
        })
    }

    fn parse_call(&mut self, callee: Expr) -> Result<Call, ParseError> {
        let start_span = callee.span;

        self.expect(TokenKind::OpenParen)?;

        let mut args = Vec::new();
        if self.current.kind != TokenKind::CloseParen {
            loop {
                args.push(self.parse_expression()?);
                if self.current.kind == TokenKind::Comma {
                    self.advance()?;
                } else {
                    break;
                }
            }
        }

        self.expect(TokenKind::CloseParen)?;

        Ok(Call {
            callee: Box::new(callee),
            args,
            span: start_span,
        })
    }

    fn parse_if(&mut self) -> Result<If, ParseError> {
        let if_span = self.current.span;
        self.expect(TokenKind::If)?;

        let cond = Box::new(self.parse_expression()?);
        let then_body = Box::new(self.parse_block()?);
        let else_body = if self.eat(TokenKind::Else)? {
            if self.current.kind == TokenKind::If {
                let else_if_span = self.current.span;
                let else_if = self.parse_if()?;
                Some(Box::new(Expr {
                    ty: (),
                    kind: ExprKind::If(else_if),
                    span: else_if_span,
                }))
            } else {
                let else_span = self.current.span;
                let block = self.parse_block()?;
                Some(Box::new(Expr {
                    ty: (),
                    kind: ExprKind::Block(block),
                    span: else_span,
                }))
            }
        } else {
            None
        };

        Ok(If {
            cond,
            then_body,
            else_body,
            span: if_span,
        })
    }

    fn parse_while(&mut self) -> Result<While, ParseError> {
        let while_span = self.current.span;
        self.expect(TokenKind::While)?;

        let cond = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_block()?);

        Ok(While {
            cond,
            body,
            span: while_span,
        })
    }

    fn prefix_binding_power(&self, token: &TokenKind) -> Option<u8> {
        match token {
            TokenKind::Minus => Some(90), // -
            TokenKind::Bang => Some(90),  // !
            _ => None,
        }
    }

    fn infix_binding_power(&self, token: &TokenKind) -> Option<(u8, u8)> {
        let bp = match token {
            TokenKind::OrOr => (10, 11),                 // ||
            TokenKind::AndAnd => (20, 21),               // &&
            TokenKind::Or => (30, 31),                   // |
            TokenKind::And => (40, 41),                  // &
            TokenKind::EqEq | TokenKind::Ne => (50, 51), // == !=
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => (60, 61), // < <= > >=
            TokenKind::Plus | TokenKind::Minus => (70, 71), // + -
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (80, 81), // * / %
            _ => return None,
        };
        Some(bp)
    }

    fn postfix_binding_power(&self, token: &TokenKind) -> Option<u8> {
        match token {
            TokenKind::OpenParen => Some(100),   // function call
            TokenKind::OpenBracket => Some(100), // array index
            _ => None,
        }
    }
}
