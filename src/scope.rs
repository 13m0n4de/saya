use std::collections::HashMap;

use crate::types::TypeId;

#[derive(Debug, Clone)]
pub enum ScopeObject {
    Var(TypeId),
    Type(TypeId),
    Function(Vec<TypeId>, TypeId),
    Const(TypeId),
    Static(TypeId),
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
