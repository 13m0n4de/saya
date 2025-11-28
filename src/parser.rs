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

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    pub fn new(mut lexer: Lexer<'a>) -> Result<Self, ParseError> {
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    pub fn parse(&mut self) -> Result<Program, ParseError> {
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

    fn parse_type(&mut self) -> Result<Ty, ParseError> {
        let type_span = self.current.span;
        let name = self.expect_identifier()?;
        match name.as_str() {
            "i64" => Ok(Ty::I64),
            _ => Err(ParseError::new(format!("Unknown type: {name}"), type_span)),
        }
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
            } else if matches!(
                expr.kind,
                ExprKind::Block(_) | ExprKind::If(_) | ExprKind::While(_)
            ) {
                // `Block`, `If`, `While` can omit semicolons after `{ }`
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
                            kind: ExprKind::Call(self.parse_call(lhs)?),
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
        let then_block = Box::new(self.parse_block()?);
        let else_block = if self.eat(TokenKind::Else)? {
            if self.current.kind == TokenKind::If {
                let else_if = self.parse_if()?;
                Some(Box::new(Expr {
                    kind: ExprKind::If(else_if),
                    span: self.current.span,
                }))
            } else {
                let block = self.parse_block()?;
                Some(Box::new(Expr {
                    kind: ExprKind::Block(block),
                    span: self.current.span,
                }))
            }
        } else {
            None
        };

        Ok(If {
            cond,
            then_block,
            else_block,
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
            TokenKind::OpenParen => Some(100),
            _ => None,
        }
    }
}
