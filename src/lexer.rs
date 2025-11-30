use std::{error::Error, fmt, str::Chars};

use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Fn,       // fn
    Return,   // return
    Let,      // let
    If,       // if
    Else,     // else
    While,    // while
    Break,    // break
    Continue, // continue
    Const,    // const
    Static,   // static

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

    OpenParen,    // (
    CloseParen,   // )
    OpenBrace,    // {
    CloseBrace,   // }
    OpenBracket,  // [
    CloseBracket, // ]
    Comma,        // ,
    Colon,        // :
    Semi,         // ;
    Arrow,        // ->

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
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

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "lex error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl Error for LexError {}

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

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace();

        self.start_span = self.span;

        let kind = match self.current {
            Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => self.read_identifier(),
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
            Some('[') => {
                self.advance();
                TokenKind::OpenBracket
            }
            Some(']') => {
                self.advance();
                TokenKind::CloseBracket
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
                if self.current == Some('/') {
                    self.advance();
                    while matches!(self.current, Some(ch) if ch != '\n') {
                        self.advance();
                    }
                    return self.next_token();
                }
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

    fn read_identifier(&mut self) -> TokenKind {
        let mut ident = String::new();

        while let Some(ch) = self.current {
            if ch.is_ascii_alphanumeric() || ch == '_' {
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
            "while" => TokenKind::While,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "const" => TokenKind::Const,
            "static" => TokenKind::Static,
            _ => TokenKind::Ident(ident),
        }
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
}
