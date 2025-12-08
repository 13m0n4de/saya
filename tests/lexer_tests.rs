use saya::lexer::{LexError, Lexer, TokenKind};

fn tokenize(input: &str) -> Result<Vec<TokenKind>, LexError> {
    let mut lexer = Lexer::new(input);
    let mut tokens = Vec::new();

    loop {
        let token = lexer.next_token()?;
        if token.kind == TokenKind::Eof {
            tokens.push(TokenKind::Eof);
            break;
        }
        tokens.push(token.kind);
    }

    Ok(tokens)
}

#[test]
fn test_keywords() {
    let input = "fn let return if else while break continue const static true false";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Fn,
            TokenKind::Let,
            TokenKind::Return,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::While,
            TokenKind::Break,
            TokenKind::Continue,
            TokenKind::Const,
            TokenKind::Static,
            TokenKind::True,
            TokenKind::False,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_identifiers_and_integers() {
    let input = "x y123 _foo 42 0 9999";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Ident("x".to_string()),
            TokenKind::Ident("y123".to_string()),
            TokenKind::Ident("_foo".to_string()),
            TokenKind::Integer(42),
            TokenKind::Integer(0),
            TokenKind::Integer(9999),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_operators() {
    let input = "+ - * / % < <= > >= == != && || ! & | =";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::Lt,
            TokenKind::Le,
            TokenKind::Gt,
            TokenKind::Ge,
            TokenKind::EqEq,
            TokenKind::Ne,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Bang,
            TokenKind::And,
            TokenKind::Or,
            TokenKind::Eq,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_double_char_operators() {
    let input = "<= >= == != && || ->";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Le,
            TokenKind::Ge,
            TokenKind::EqEq,
            TokenKind::Ne,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Arrow,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_delimiters() {
    let input = "( ) { } [ ] , : ; ->";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::OpenParen,
            TokenKind::CloseParen,
            TokenKind::OpenBrace,
            TokenKind::CloseBrace,
            TokenKind::OpenBracket,
            TokenKind::CloseBracket,
            TokenKind::Comma,
            TokenKind::Colon,
            TokenKind::Semi,
            TokenKind::Arrow,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_string_literal() {
    let input = r#""hello" "world 123" "with\nescape""#;
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::String("hello".to_string()),
            TokenKind::String("world 123".to_string()),
            TokenKind::String("with\\nescape".to_string()),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_compact_code() {
    let input = "let x=42;";
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Let,
            TokenKind::Ident("x".to_string()),
            TokenKind::Eq,
            TokenKind::Integer(42),
            TokenKind::Semi,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_comments() {
    let input = r#"
        // this is a comment
        let x = 42; // another comment
        // final comment
    "#;
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Let,
            TokenKind::Ident("x".to_string()),
            TokenKind::Eq,
            TokenKind::Integer(42),
            TokenKind::Semi,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_complete_function() {
    let input = r#"
        fn add(a: i64, b: i64) -> i64 {
            return a + b;
        }
    "#;
    let tokens = tokenize(input).unwrap();

    assert_eq!(
        tokens,
        vec![
            TokenKind::Fn,
            TokenKind::Ident("add".to_string()),
            TokenKind::OpenParen,
            TokenKind::Ident("a".to_string()),
            TokenKind::Colon,
            TokenKind::Ident("i64".to_string()),
            TokenKind::Comma,
            TokenKind::Ident("b".to_string()),
            TokenKind::Colon,
            TokenKind::Ident("i64".to_string()),
            TokenKind::CloseParen,
            TokenKind::Arrow,
            TokenKind::Ident("i64".to_string()),
            TokenKind::OpenBrace,
            TokenKind::Return,
            TokenKind::Ident("a".to_string()),
            TokenKind::Plus,
            TokenKind::Ident("b".to_string()),
            TokenKind::Semi,
            TokenKind::CloseBrace,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn test_invalid_character() {
    let input = "let x = @";
    let result = tokenize(input);

    assert!(result.is_err(), "should fail on invalid character");
}

#[test]
fn test_unterminated_string() {
    let input = r#"let x = "hello"#;
    let result = tokenize(input);

    assert!(result.is_err(), "should fail on unterminated string");
}

#[test]
fn test_string_with_newline() {
    let input = "let x = \"hello\nworld\"";
    let result = tokenize(input);

    assert!(result.is_err(), "should fail on string with newline");
}

#[test]
fn test_integer_overflow() {
    let input = "99999999999999999999";
    let result = tokenize(input);

    assert!(result.is_err(), "should fail on integer overflow");
}

#[test]
fn test_empty_input() {
    let tokens = tokenize("").unwrap();
    assert_eq!(tokens, vec![TokenKind::Eof]);
}

#[test]
fn test_only_whitespace() {
    let tokens = tokenize("   \t\n  ").unwrap();
    assert_eq!(tokens, vec![TokenKind::Eof]);
}

#[test]
fn test_only_comments() {
    let tokens = tokenize("// just a comment\n// another one").unwrap();
    assert_eq!(tokens, vec![TokenKind::Eof]);
}
