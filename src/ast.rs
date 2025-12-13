use crate::{span::Span, ty::Type};

#[derive(Debug, Clone, PartialEq)]
pub struct Program<T = ()> {
    pub items: Vec<Item<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item<T = ()> {
    Const(ConstDef<T>),
    Static(StaticDef<T>),
    Function(FunctionDef<T>),
    Struct(StructDef),
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
    pub type_ann: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunctionDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type_ann: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef<T = ()> {
    pub name: String,
    pub type_ann: Type,
    pub init: Box<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef<T = ()> {
    pub name: String,
    pub type_ann: Type,
    pub init: Box<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub type_ann: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<T = ()> {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type_ann: Type,
    pub body: Block<T>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_ann: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block<T = ()> {
    pub stmts: Vec<Stmt<T>>,
    pub ty: T,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt<T = ()> {
    pub kind: StmtKind<T>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind<T = ()> {
    Expr(Expr<T>),
    Semi(Expr<T>),
    Let(Let<T>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Let<T = ()> {
    pub name: String,
    pub type_ann: Type,
    pub init: Expr<T>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr<T = ()> {
    pub kind: ExprKind<T>,
    pub ty: T,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind<T = ()> {
    Literal(Literal),
    Struct(StructExpr<T>),
    Ident(String),
    Array(Vec<Expr<T>>),
    Repeat(Box<Expr<T>>, Box<Expr<T>>),
    Index(Box<Expr<T>>, Box<Expr<T>>),
    Call(Call<T>),
    Unary(UnaryOp, Box<Expr<T>>),
    Binary(BinaryOp, Box<Expr<T>>, Box<Expr<T>>),
    Assign(Box<Expr<T>>, Box<Expr<T>>),
    Return(Option<Box<Expr<T>>>),
    Block(Block<T>),
    If(If<T>),
    While(While<T>),
    Break,
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructExpr<T> {
    pub name: String,
    pub fields: Vec<FieldInit<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit<T = ()> {
    pub name: String,
    pub value: Box<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call<T = ()> {
    pub callee: Box<Expr<T>>,
    pub args: Vec<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct If<T = ()> {
    pub cond: Box<Expr<T>>,
    pub then_body: Box<Block<T>>,
    pub else_body: Option<Box<Expr<T>>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct While<T = ()> {
    pub cond: Box<Expr<T>>,
    pub body: Box<Block<T>>,
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
