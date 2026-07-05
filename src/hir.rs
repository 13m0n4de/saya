use crate::{ast, span::Span, types::TypeId};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub uses: Vec<String>,
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
    TypeDef(TypeDef),
    TypeAlias(TypeAlias),
    Extern(ExternItem),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub ident: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeAlias {
    pub ident: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternItem {
    Static(ExternStaticDecl),
    Function(ExternFunctionDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternStaticDecl {
    pub ident: String,
    pub symbol: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunctionDecl {
    pub ident: String,
    pub symbol: String,
    pub params: Vec<Param>,
    pub is_variadic: bool,
    pub return_type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub ident: String,
    pub init: ConstVal,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef {
    pub ident: String,
    pub symbol: String,
    pub init: ConstVal,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub ident: String,
    pub symbol: String,
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
    Const(ConstVal),
    Struct(StructExpr),
    Place(Place),
    Array(Vec<Expr>),
    Repeat(Box<Expr>, usize),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Call(Call),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, TypeId),
    Assign(Box<Expr>, Box<Expr>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(If),
    While(While),
    Loop(Loop),
    Break(Option<Box<Expr>>),
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(String),
    CString(String),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstVal {
    pub kind: ConstValKind,
    pub type_id: TypeId,
}

impl ConstVal {
    pub fn is_zero(&self) -> bool {
        match &self.kind {
            ConstValKind::Integer(n) => *n == 0,
            ConstValKind::Float(n) => *n == 0.0,
            ConstValKind::Bool(b) => !b,
            ConstValKind::Struct(fields) => fields.iter().all(Self::is_zero),
            ConstValKind::Array(elems) => elems.iter().all(Self::is_zero),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValKind {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(String),
    CString(String),
    Null,
    Struct(Vec<ConstVal>),
    Array(Vec<ConstVal>),
    Repeat(Box<ConstVal>, usize),
}

impl ConstVal {
    pub fn new(kind: ConstValKind, type_id: TypeId) -> Self {
        Self { kind, type_id }
    }
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
    pub variadic_start: Option<u64>,
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
pub struct Loop {
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
