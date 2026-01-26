use std::{collections::HashMap, rc::Rc};

use crate::{ast, hir, types::TypeId};

pub enum Scope {
    Module {
        objects: HashMap<String, ScopeObject>,
    },
    Function {
        return_type_id: TypeId,
        objects: HashMap<String, ScopeObject>,
    },
    Loop {
        objects: HashMap<String, ScopeObject>,
    },
    Block {
        objects: HashMap<String, ScopeObject>,
    },
}

impl Scope {
    pub fn get(&self, name: &str) -> Option<&ScopeObject> {
        self.objects().get(name)
    }

    pub fn insert(&mut self, name: String, object: ScopeObject) -> Option<ScopeObject> {
        self.objects_mut().insert(name, object)
    }

    pub fn objects(&self) -> &HashMap<String, ScopeObject> {
        match self {
            Self::Module { objects }
            | Self::Function { objects, .. }
            | Self::Loop { objects }
            | Self::Block { objects } => objects,
        }
    }

    pub fn objects_mut(&mut self) -> &mut HashMap<String, ScopeObject> {
        match self {
            Self::Module { objects }
            | Self::Function { objects, .. }
            | Self::Loop { objects }
            | Self::Block { objects } => objects,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ScopeObject {
    Var(TypeId),
    Const(Const),
    Static(Static),
    Function(Function),
    Struct(Struct),
    ExternStatic(ExternStatic),
    ExternFunction(ExternFunction),
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

#[derive(Debug, Clone)]
pub enum ExternStatic {
    Unresolved(Rc<ast::ExternStaticDecl>),
    Resolved(TypeId),
}

#[derive(Debug, Clone)]
pub enum ExternFunction {
    Unresolved(Rc<ast::ExternFunctionDecl>),
    Resolved(Vec<TypeId>, TypeId),
}
