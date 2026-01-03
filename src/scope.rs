use std::{collections::HashMap, rc::Rc};

use crate::{ast, hir, types::TypeId};

#[derive(Debug, Clone)]
pub enum ScopeObject {
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

#[derive(Default)]
pub struct Scope {
    objects: HashMap<String, ScopeObject>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, obj: ScopeObject) -> Option<ScopeObject> {
        self.objects.insert(name, obj)
    }

    pub fn get(&self, name: &str) -> Option<&ScopeObject> {
        self.objects.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &ScopeObject)> {
        self.objects.iter()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.objects.contains_key(name)
    }
}
