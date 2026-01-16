use std::{collections::HashMap, rc::Rc};

use crate::{ast, hir, types::TypeId};

pub type Scope = HashMap<String, ScopeObject>;

#[derive(Debug, Clone)]
pub struct ScopeObject {
    pub vis: ast::Visibility,
    pub kind: ScopeKind,
}

impl ScopeObject {
    pub fn new(vis: ast::Visibility, kind: ScopeKind) -> Self {
        Self { vis, kind }
    }

    pub fn private(kind: ScopeKind) -> Self {
        Self {
            vis: ast::Visibility::Private,
            kind,
        }
    }

    pub fn public(kind: ScopeKind) -> Self {
        Self {
            vis: ast::Visibility::Public,
            kind,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ScopeKind {
    Var(TypeId),
    Const(Const),
    Static(Static),
    Function(Function),
    Struct(Struct),
}

#[derive(Debug, Clone)]
pub enum Const {
    Unresolved(Rc<ast::ConstDef>),
    Resolving(Rc<ast::ConstDef>),
    Resolved(TypeId, hir::Literal),
}

#[derive(Debug, Clone)]
pub enum Static {
    Unresolved(Rc<ast::StaticDef>),
    Resolving(Rc<ast::StaticDef>),
    Resolved(TypeId, hir::Literal),
}

#[derive(Debug, Clone)]
pub enum Function {
    Unresolved(Rc<ast::FunctionDef>),
    Resolving(Rc<ast::FunctionDef>),
    Resolved(Vec<TypeId>, TypeId),
}

#[derive(Debug, Clone)]
pub enum Struct {
    Unresolved(Rc<ast::StructDef>),
    Resolving(Rc<ast::StructDef>),
    Resolved(TypeId),
}
