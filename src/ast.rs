use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I64,
    U8,
    Bool,
    Pointer(Box<Type>),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Unit,
    Never,
}

impl Type {
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Type::Slice(_) | Type::Array(..))
    }

    pub fn size(&self) -> usize {
        match self {
            Type::I64 => 8,
            Type::U8 => 1,
            Type::Bool => 1,
            Type::Pointer(_) => 8,
            Type::Slice(_) => 16, // { ptr: l, len: l }
            Type::Array(elem_ty, count) => elem_ty.size() * count,
            Type::Unit => 0,
            Type::Never => 0,
        }
    }

    pub fn align(&self) -> u64 {
        match self {
            Type::I64 => 8,
            Type::U8 => 1,
            Type::Bool => 1,
            Type::Pointer(_) => 8,
            Type::Slice(_) => 8,
            Type::Array(elem_ty, _) => elem_ty.align(),
            Type::Unit => 1,
            Type::Never => 1,
        }
    }

    pub fn to_qbe_base(&self) -> qbe::Type<'static> {
        match self {
            Type::I64 => qbe::Type::Long,
            Type::U8 => qbe::Type::Word,
            Type::Bool => qbe::Type::Word,
            Type::Pointer(_) => qbe::Type::Long,
            Type::Array(_, _) => qbe::Type::Long,
            Type::Slice(_) => qbe::Type::Long,
            Type::Unit => qbe::Type::Long,
            Type::Never => qbe::Type::Long,
        }
    }

    pub fn to_qbe_load(&self) -> qbe::Type<'static> {
        match self {
            Type::I64 => qbe::Type::Long,
            Type::U8 => qbe::Type::UnsignedByte,
            Type::Bool => qbe::Type::UnsignedByte,
            Type::Pointer(_) => qbe::Type::Long,
            Type::Array(_, _) => qbe::Type::Long,
            Type::Slice(_) => qbe::Type::Long,
            Type::Unit => qbe::Type::Long,
            Type::Never => qbe::Type::Long,
        }
    }

    pub fn to_qbe_store(&self) -> qbe::Type<'static> {
        match self {
            Type::I64 => qbe::Type::Long,
            Type::U8 => qbe::Type::Byte,
            Type::Bool => qbe::Type::Byte,
            Type::Pointer(_) => qbe::Type::Long,
            Type::Array(_, _) => qbe::Type::Long,
            Type::Slice(_) => qbe::Type::Long,
            Type::Unit => qbe::Type::Long,
            Type::Never => qbe::Type::Long,
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
