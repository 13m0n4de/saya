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

    fn parse_type_ann(&mut self) -> Result<TypeAnn, ParseError> {
        let start_span = self.current.span;

        let kind = match self.current.kind.clone() {
            // Base types: i64, u8, bool
            TokenKind::Ident(name) => {
                self.advance()?;
                match name.as_str() {
                    "i64" => TypeAnnKind::I64,
                    "u8" => TypeAnnKind::U8,
                    "bool" => TypeAnnKind::Bool,
                    name => TypeAnnKind::Named(name.to_string()),
                }
            }
            // Slice: [T] or Array: [T; N]
            TokenKind::OpenBracket => {
                self.advance()?;
                let elem_type_ann = self.parse_type_ann()?;

                match self.current.kind {
                    // Slice: [T]
                    TokenKind::CloseBracket => {
                        self.advance()?;
                        TypeAnnKind::Slice(Box::new(elem_type_ann))
                    }
                    // Array: [T; N]
                    TokenKind::Semi => {
                        self.advance()?;
                        let count_expr = self.parse_expression()?;
                        self.expect(TokenKind::CloseBracket)?;
                        TypeAnnKind::Array(Box::new(elem_type_ann), Box::new(count_expr))
                    }
                    _ => {
                        return Err(ParseError::new(
                            format!(
                                "Expected `]` or `;` after type in brackets, found {:?}",
                                self.current.kind
                            ),
                            self.current.span,
                        ));
                    }
                }
            }
            // Pointer: *T
            TokenKind::Star => {
                self.advance()?;
                let inner_type = self.parse_type_ann()?;
                TypeAnnKind::Pointer(Box::new(inner_type))
            }
            // Unit: ()
            TokenKind::OpenParen => {
                self.advance()?;
                self.expect(TokenKind::CloseParen)?;
                TypeAnnKind::Unit
            }
            // Never: !
            TokenKind::Bang => {
                self.advance()?;
                TypeAnnKind::Never
            }
            _ => {
                return Err(ParseError::new(
                    format!("Unknown type: {:?}", self.current.kind),
                    self.current.span,
                ));
            }
        };

        Ok(TypeAnn {
            kind,
            span: start_span,
        })
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();

        loop {
            match self.current.kind {
                TokenKind::Const => items.push(Item::Const(self.parse_const()?)),
                TokenKind::Static => items.push(Item::Static(self.parse_static()?)),
                TokenKind::Struct => items.push(Item::Struct(self.parse_struct()?)),
                TokenKind::Fn => items.push(Item::Function(self.parse_function()?)),
                TokenKind::Extern => items.push(self.parse_item_extern()?),
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

    fn parse_item_extern(&mut self) -> Result<Item, ParseError> {
        self.expect(TokenKind::Extern)?;

        match self.current.kind {
            TokenKind::Fn => Ok(Item::Extern(ExternItem::Function(
                self.parse_extern_function()?,
            ))),
            TokenKind::Static => Ok(Item::Extern(ExternItem::Static(
                self.parse_extern_static()?,
            ))),
            _ => Err(ParseError::new(
                "expected 'fn' or 'static' after 'extern'".to_string(),
                self.current.span,
            )),
        }
    }

    fn parse_extern_static(&mut self) -> Result<ExternStaticDecl, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Static)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::Colon)?;

        let type_ann = self.parse_type_ann()?;

        self.expect(TokenKind::Semi)?;

        Ok(ExternStaticDecl {
            name,
            type_ann,
            span: start_span,
        })
    }

    fn parse_extern_function(&mut self) -> Result<ExternFunctionDecl, ParseError> {
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
            TypeAnn {
                kind: TypeAnnKind::Unit,
                span: self.current.span,
            }
        };

        self.expect(TokenKind::Semi)?;

        Ok(ExternFunctionDecl {
            name,
            params,
            return_type_ann,
            span: start_span,
        })
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

    fn parse_struct(&mut self) -> Result<StructDef, ParseError> {
        let start_span = self.current.span;

        self.expect(TokenKind::Struct)?;

        let name = self.expect_identifier()?;

        self.expect(TokenKind::OpenBrace)?;

        let fields = if self.current.kind == TokenKind::CloseBrace {
            Vec::new()
        } else {
            self.parse_field_list()?
        };

        self.expect(TokenKind::CloseBrace)?;

        Ok(StructDef {
            name,
            fields,
            span: start_span,
        })
    }

    fn parse_field_list(&mut self) -> Result<Vec<Field>, ParseError> {
        let mut fields = Vec::new();

        fields.push(self.parse_field()?);

        while self.eat(TokenKind::Comma)? {
            fields.push(self.parse_field()?);
        }

        Ok(fields)
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        let start_span = self.current.span;

        let name = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let type_ann = self.parse_type_ann()?;

        Ok(Field {
            name,
            type_ann,
            span: start_span,
        })
    }

    fn parse_field_init_list(&mut self) -> Result<Vec<FieldInit>, ParseError> {
        let mut fields = Vec::new();

        fields.push(self.parse_field_init()?);

        while self.eat(TokenKind::Comma)? {
            fields.push(self.parse_field_init()?);
        }

        Ok(fields)
    }

    fn parse_field_init(&mut self) -> Result<FieldInit, ParseError> {
        let start_span = self.current.span;

        let name = self.expect_identifier()?;
        self.expect(TokenKind::Colon)?;
        let value = Box::new(self.parse_expression()?);

        Ok(FieldInit {
            name,
            value,
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
            TypeAnn {
                kind: TypeAnnKind::Unit,
                span: self.current.span,
            }
        };

        let body = self.parse_block()?;

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

        while self.current.kind != TokenKind::CloseBrace {
            stmts.push(self.parse_statement()?);
        }

        self.expect(TokenKind::CloseBrace)?;

        Ok(Block {
            stmts,
            span: start_span,
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        if self.current.kind == TokenKind::Let {
            self.parse_stmt_let()
        } else {
            let expr = self.parse_expression()?;
            let expr_span = expr.span;

            if self.eat(TokenKind::Semi)? {
                // Explicit semicolon: statement
                Ok(Stmt {
                    kind: StmtKind::Semi(expr),
                    span: expr_span,
                })
            } else if self.current.kind == TokenKind::CloseBrace {
                // No semicolon, followed by '}': tail expression
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: expr_span,
                })
            } else if matches!(
                expr.kind,
                ExprKind::Block(_) | ExprKind::If(_) | ExprKind::While(_)
            ) {
                // `Block`, `If`, `While` can omit semicolons after `{ }`
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    span: expr_span,
                })
            } else {
                // Other expressions require semicolons
                Err(ParseError::new(
                    format!(
                        "Expected ';' after expression (found {:?})",
                        self.current.kind
                    ),
                    self.current.span,
                ))
            }
        }
    }

    fn parse_stmt_let(&mut self) -> Result<Stmt, ParseError> {
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

    pub fn parse_expression(&mut self) -> Result<Expr, ParseError> {
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
        let start_span = lhs.span;

        if self.current.kind == TokenKind::Eq {
            self.advance()?;
            let rhs = self.parse_expr_assign()?;

            Ok(Expr {
                kind: ExprKind::Assign(Box::new(lhs), Box::new(rhs)),
                span: start_span,
            })
        } else {
            Ok(lhs)
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = if let Some(prefix_bp) = Self::prefix_binding_power(&self.current.kind) {
            // Prefix operator
            let start_span = self.current.span;
            let op = match self.current.kind {
                TokenKind::Minus => UnaryOp::Neg,
                TokenKind::Bang => UnaryOp::Not,
                TokenKind::And => UnaryOp::Ref,
                TokenKind::Star => UnaryOp::Deref,
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
            let primary = self.parse_expr_primary()?;

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
            if let Some(postfix_bp) = Self::postfix_binding_power(op_token) {
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
                    TokenKind::OpenBracket => {
                        let span = lhs.span;
                        self.advance()?;
                        let index = self.parse_expression()?;
                        self.expect(TokenKind::CloseBracket)?;
                        lhs = Expr {
                            kind: ExprKind::Index(Box::new(lhs), Box::new(index)),
                            span,
                        }
                    }
                    TokenKind::Dot => {
                        let span = lhs.span;
                        self.advance()?;
                        let field_name = self.expect_identifier()?;
                        lhs = Expr {
                            kind: ExprKind::Field(Box::new(lhs), field_name),
                            span,
                        }
                    }
                    _ => unreachable!(),
                }

                continue;
            }

            // Infix operators
            if let Some((left_bp, right_bp)) = Self::infix_binding_power(op_token) {
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

    fn parse_expr_primary(&mut self) -> Result<Expr, ParseError> {
        let start_span = self.current.span;

        let kind = match &self.current.kind {
            TokenKind::Integer(val, suffix) => {
                let val = *val;
                let suffix = suffix.clone();
                self.advance()?;
                ExprKind::Literal(Literal::Integer(val, suffix))
            }
            TokenKind::String(str) => {
                let val = str.to_owned();
                self.advance()?;
                ExprKind::Literal(Literal::String(val))
            }
            TokenKind::CString(str) => {
                let val = str.to_owned();
                self.advance()?;
                ExprKind::Literal(Literal::CString(val))
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

                if self.eat(TokenKind::OpenBrace)? {
                    let fields = if self.current.kind == TokenKind::CloseBrace {
                        Vec::new()
                    } else {
                        self.parse_field_init_list()?
                    };
                    self.expect(TokenKind::CloseBrace)?;

                    ExprKind::Struct(StructExpr {
                        name,
                        fields,
                        span: start_span,
                    })
                } else {
                    ExprKind::Ident(name)
                }
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

    fn parse_expr_cond(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.current.kind, TokenKind::Ident(_)) {
            let next = self.lexer.peek_token()?;
            if next.kind == TokenKind::OpenBrace {
                let span = self.current.span;
                let name = self.expect_identifier()?;
                return Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span,
                });
            }
        }

        self.parse_expression()
    }

    fn parse_if(&mut self) -> Result<If, ParseError> {
        let if_span = self.current.span;
        self.expect(TokenKind::If)?;

        let cond = Box::new(self.parse_expr_cond()?);
        let then_body = Box::new(self.parse_block()?);
        let else_body = if self.eat(TokenKind::Else)? {
            if self.current.kind == TokenKind::If {
                let else_if_span = self.current.span;
                let else_if = self.parse_if()?;
                Some(Box::new(Expr {
                    kind: ExprKind::If(else_if),
                    span: else_if_span,
                }))
            } else {
                let else_span = self.current.span;
                let block = self.parse_block()?;
                Some(Box::new(Expr {
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

        let cond = Box::new(self.parse_expr_cond()?);
        let body = Box::new(self.parse_block()?);

        Ok(While {
            cond,
            body,
            span: while_span,
        })
    }

    fn prefix_binding_power(token: &TokenKind) -> Option<u8> {
        match token {
            TokenKind::Minus => Some(90), // -
            TokenKind::Bang => Some(90),  // !
            TokenKind::And => Some(90),   // &
            TokenKind::Star => Some(90),  // *
            _ => None,
        }
    }

    fn infix_binding_power(token: &TokenKind) -> Option<(u8, u8)> {
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

    fn postfix_binding_power(token: &TokenKind) -> Option<u8> {
        match token {
            TokenKind::OpenParen => Some(100),   // function call
            TokenKind::OpenBracket => Some(100), // array index
            TokenKind::Dot => Some(100),         // field
            _ => None,
        }
    }
}
