use std::{collections::HashMap, rc::Rc};

use crate::{ast, hir, types::TypeId};

#[derive(Default)]
pub struct Scopes {
    stack: Vec<Scope>,
}

impl Scopes {
    pub fn new() -> Self {
        Self { stack: vec![] }
    }

    pub fn push(&mut self, scope: Scope) {
        self.stack.push(scope);
    }

    pub fn pop(&mut self) {
        self.stack
            .pop()
            .expect("ICE: cannot pop scope, scopes stack is empty");
    }

    pub fn last_mut(&mut self) -> &mut Scope {
        self.stack
            .last_mut()
            .expect("scope stack should not be empty")
    }

    pub fn first_mut(&mut self) -> &mut Scope {
        self.stack
            .first_mut()
            .expect("scope stack should not be empty")
    }

    pub fn find<P>(&self, predicate: P) -> Option<&Scope>
    where
        P: Fn(&Scope) -> bool,
    {
        self.stack.iter().rev().find(|s| predicate(s))
    }

    pub fn find_map<T, F>(&self, f: F) -> Option<&T>
    where
        F: Fn(&Scope) -> Option<&T>,
    {
        self.stack.iter().rev().find_map(f)
    }

    pub fn lookup(&self, name: &str) -> Option<&ScopeObject> {
        self.stack.iter().rev().find_map(|s| s.get(name))
    }
}

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

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, ScopeObject)>,
    {
        self.objects_mut().extend(iter);
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
