use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnn {
    I64,
    Str,
    Array(Box<TypeAnn>, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I64,
    Str,
    Array(Box<Type>, usize),
    Unit,
    Never,
}

impl From<&TypeAnn> for Type {
    fn from(ty: &TypeAnn) -> Self {
        match ty {
            TypeAnn::I64 => Type::I64,
            TypeAnn::Str => Type::Str,
            TypeAnn::Array(elem, size) => Type::Array(Box::new(Type::from(elem.as_ref())), *size),
        }
    }
}

impl From<&Type> for qbe::Type<'static> {
    fn from(ty: &Type) -> Self {
        match ty {
            Type::I64 => qbe::Type::Long,
            Type::Str => qbe::Type::Long,
            Type::Array(_, _) => qbe::Type::Long,
            Type::Unit => qbe::Type::Zero,
            Type::Never => unreachable!("Never type should not need QBE type conversion"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program<T = ()> {
    pub items: Vec<Item<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item<T = ()> {
    Const(ConstDef<T>),
    Static(StaticDef<T>),
    Function(FunctionDef<T>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef<T = ()> {
    pub name: String,
    pub type_ann: TypeAnn,
    pub init: Box<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef<T = ()> {
    pub name: String,
    pub type_ann: TypeAnn,
    pub init: Box<Expr<T>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<T = ()> {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type_ann: TypeAnn,
    pub body: Block<T>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_ann: TypeAnn,
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
    pub type_ann: TypeAnn,
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
