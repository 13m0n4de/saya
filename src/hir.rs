use crate::{ast, span::Span, types::TypeId};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub vis: Visibility,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

impl From<&ast::Visibility> for Visibility {
    fn from(value: &ast::Visibility) -> Self {
        match value {
            ast::Visibility::Public => Self::Public,
            ast::Visibility::Private => Self::Private,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Const(ConstDef),
    Static(StaticDef),
    Function(FunctionDef),
    Extern(ExternItem),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternItem {
    Static(ExternStaticDecl),
    Function(ExternFunctionDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternStaticDecl {
    pub name: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunctionDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub ident: String,
    pub type_id: TypeId,
    pub init: Literal,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef {
    pub ident: String,
    pub type_id: TypeId,
    pub init: Literal,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub ident: String,
    pub params: Vec<Param>,
    pub return_type_id: TypeId,
    pub body: Option<Block>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub type_id: TypeId,
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
    pub type_id: TypeId,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Place {
    Local(String),
    Global(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    Struct(StructExpr),
    Place(Place),
    Array(Vec<Expr>),
    Repeat(Box<Expr>, Literal),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Call(Call),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(If),
    While(While),
    Break,
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Bool(bool),
    String(String),
    CString(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructExpr {
    pub ident: String,
    pub fields: Vec<FieldInit>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    pub name: String,
    pub value: Box<Expr>,
    pub span: Span,
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
    pub then_body: Box<Block>,
    pub else_body: Option<Box<Expr>>,
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
    Neg,   // -
    Not,   // !
    Ref,   // &
    Deref, // *
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

impl From<&ast::UnaryOp> for UnaryOp {
    fn from(value: &ast::UnaryOp) -> Self {
        match value {
            ast::UnaryOp::Neg => Self::Neg,
            ast::UnaryOp::Not => Self::Not,
            ast::UnaryOp::Ref => Self::Ref,
            ast::UnaryOp::Deref => Self::Deref,
        }
    }
}

impl From<&ast::BinaryOp> for BinaryOp {
    fn from(value: &ast::BinaryOp) -> Self {
        match value {
            ast::BinaryOp::Add => Self::Add,
            ast::BinaryOp::Sub => Self::Sub,
            ast::BinaryOp::Mul => Self::Mul,
            ast::BinaryOp::Div => Self::Div,
            ast::BinaryOp::Rem => Self::Rem,
            ast::BinaryOp::Lt => Self::Lt,
            ast::BinaryOp::Le => Self::Le,
            ast::BinaryOp::Gt => Self::Gt,
            ast::BinaryOp::Ge => Self::Ge,
            ast::BinaryOp::Eq => Self::Eq,
            ast::BinaryOp::Ne => Self::Ne,
            ast::BinaryOp::BitAnd => Self::BitAnd,
            ast::BinaryOp::BitOr => Self::BitOr,
            ast::BinaryOp::And => Self::And,
            ast::BinaryOp::Or => Self::Or,
        }
    }
}
