use std::fmt;

use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Pub,      // pub
    Use,      // use
    As,       // as
    Fn,       // fn
    Extern,   // extern
    Return,   // return
    Struct,   // struct
    Type,     // type
    Let,      // let
    If,       // if
    Else,     // else
    While,    // while
    Loop,     // loop
    Break,    // break
    Continue, // continue
    Const,    // const
    Static,   // static
    True,     // true
    False,    // false
    Some,     // some
    None,     // none

    Ident(String),
    Integer(i64, Option<String>),
    Float(f64, Option<String>),
    String(String),
    CString(String),

    Plus,    // +
    Minus,   // -
    Star,    // *
    Slash,   // /
    Percent, // %

    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=

    And,      // &
    Or,       // |
    Eq,       // =
    Bang,     // !
    Question, // ?

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
    Dot,          // .
    DotDotDot,    //...
    Comma,        // ,
    Semi,         // ;
    Colon,        // :
    PathSep,      // ::
    Arrow,        // ->
    At,           // @

    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            TokenKind::Ident(name) => return write!(f, "identifier `{name}`"),
            TokenKind::Integer(n, suffix) => {
                return write!(
                    f,
                    "integer literal `{n}{}`",
                    suffix.as_deref().unwrap_or("")
                );
            }
            TokenKind::Float(..) => return f.write_str("float literal"),
            TokenKind::String(_) => return f.write_str("string literal"),
            TokenKind::CString(_) => return f.write_str("C string literal"),
            TokenKind::Eof => return f.write_str("end of file"),

            TokenKind::Pub => "pub",
            TokenKind::Use => "use",
            TokenKind::As => "as",
            TokenKind::Fn => "fn",
            TokenKind::Extern => "extern",
            TokenKind::Return => "return",
            TokenKind::Struct => "struct",
            TokenKind::Type => "type",
            TokenKind::Let => "let",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::While => "while",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Const => "const",
            TokenKind::Static => "static",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::Some => "some",
            TokenKind::None => "none",

            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Lt => "<",
            TokenKind::Le => "<=",
            TokenKind::Gt => ">",
            TokenKind::Ge => ">=",
            TokenKind::And => "&",
            TokenKind::Or => "|",
            TokenKind::Eq => "=",
            TokenKind::Bang => "!",
            TokenKind::Question => "?",
            TokenKind::EqEq => "==",
            TokenKind::Ne => "!=",
            TokenKind::AndAnd => "&&",
            TokenKind::OrOr => "||",
            TokenKind::OpenParen => "(",
            TokenKind::CloseParen => ")",
            TokenKind::OpenBrace => "{",
            TokenKind::CloseBrace => "}",
            TokenKind::OpenBracket => "[",
            TokenKind::CloseBracket => "]",
            TokenKind::Dot => ".",
            TokenKind::DotDotDot => "...",
            TokenKind::Comma => ",",
            TokenKind::Semi => ";",
            TokenKind::Colon => ":",
            TokenKind::PathSep => "::",
            TokenKind::Arrow => "->",
            TokenKind::At => "@",
        };
        write!(f, "`{s}`")
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
