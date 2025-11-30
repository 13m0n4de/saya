use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Const(ConstDef),
    Static(StaticDef),
    Function(FunctionDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub name: String,
    pub ty: Ty,
    pub init: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef {
    pub name: String,
    pub ty: Ty,
    pub init: Box<Expr>,
    pub span: Span,
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
    Array(Box<Ty>, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
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
    Array(Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Call(Call),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Assign(String, Box<Expr>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(If),
    While(While),
    Break,
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct If {
    pub cond: Box<Expr>,
    pub then_block: Box<Block>,
    pub else_block: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct While {
    pub cond: Box<Expr>,
    pub body: Box<Block>,
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
    And,    // &&
    Or,     // ||
}
