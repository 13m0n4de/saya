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
            .expect("ICE: scope stack should not be empty")
    }

    pub fn first_mut(&mut self) -> &mut Scope {
        self.stack
            .first_mut()
            .expect("ICE: scope stack should not be empty")
    }

    pub fn find<P>(&self, predicate: P) -> Option<&Scope>
    where
        P: Fn(&Scope) -> bool,
    {
        self.stack.iter().rev().find(|s| predicate(s))
    }

    pub fn find_mut<P>(&mut self, predicate: P) -> Option<&mut Scope>
    where
        P: Fn(&Scope) -> bool,
    {
        self.stack.iter_mut().rev().find(|s| predicate(s))
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

pub struct Scope {
    pub kind: ScopeKind,
    pub objects: HashMap<String, ScopeObject>,
}

pub enum ScopeKind {
    Module,
    Function {
        return_type_id: TypeId,
    },
    Loop {
        break_type: Option<TypeId>,
        allows_break_value: bool,
    },
    Block,
}

impl Scope {
    pub fn get(&self, name: &str) -> Option<&ScopeObject> {
        self.objects.get(name)
    }

    pub fn insert(&mut self, name: String, object: ScopeObject) -> Option<ScopeObject> {
        self.objects.insert(name, object)
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, ScopeObject)>,
    {
        self.objects.extend(iter);
    }
}

#[derive(Debug, Clone)]
pub enum ScopeObject {
    Var(TypeId),
    Const(Const),
    Static(Static),
    Function(Function),
    Struct(Struct),
    TypeAlias(TypeAlias),
    ExternStatic(ExternStatic),
    ExternFunction(ExternFunction),
}

#[derive(Debug, Clone)]
pub enum Const {
    Unresolved(Rc<ast::ConstDef>),
    Resolving(Rc<ast::ConstDef>),
    Resolved(hir::ConstVal),
}

#[derive(Debug, Clone)]
pub enum Static {
    Unresolved(Rc<ast::Item>),
    Resolving(Rc<ast::Item>),
    Resolved { value: hir::ConstVal, symbol: String },
}

#[derive(Debug, Clone)]
pub enum Function {
    Unresolved(Rc<ast::Item>),
    Resolving(Rc<ast::Item>),
    Resolved {
        params: Vec<TypeId>,
        ret: TypeId,
        symbol: String,
    },
}

#[derive(Debug, Clone)]
pub enum Struct {
    Unresolved(Rc<ast::StructDef>),
    Resolving(Rc<ast::StructDef>),
    Resolved(TypeId),
}

#[derive(Debug, Clone)]
pub enum TypeAlias {
    Unresolved(Rc<ast::TypeAliasDef>),
    Resolving(Rc<ast::TypeAliasDef>),
    Resolved(TypeId),
}

#[derive(Debug, Clone)]
pub enum ExternStatic {
    Unresolved(Rc<ast::Item>),
    Resolved { type_id: TypeId, symbol: String },
}

#[derive(Debug, Clone)]
pub enum ExternFunction {
    Unresolved(Rc<ast::Item>),
    Resolved {
        params: Vec<TypeId>,
        ret: TypeId,
        is_variadic: bool,
        symbol: String,
    },
}
