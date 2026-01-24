use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);

impl TypeId {
    pub const I64: TypeId = TypeId(0);
    pub const U8: TypeId = TypeId(1);
    pub const BOOL: TypeId = TypeId(2);
    pub const UNIT: TypeId = TypeId(3);
    pub const NEVER: TypeId = TypeId(4);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub kind: TypeKind,
    pub size: usize,
    pub align: u64,
}

impl Type {
    pub fn i64() -> Self {
        Type {
            kind: TypeKind::I64,
            size: 8,
            align: 8,
        }
    }

    pub fn u8() -> Self {
        Type {
            kind: TypeKind::U8,
            size: 1,
            align: 1,
        }
    }

    pub fn bool() -> Self {
        Type {
            kind: TypeKind::Bool,
            size: 1,
            align: 1,
        }
    }

    pub fn unit() -> Self {
        Type {
            kind: TypeKind::Unit,
            size: 0,
            align: 1,
        }
    }

    pub fn never() -> Self {
        Type {
            kind: TypeKind::Never,
            size: 0,
            align: 1,
        }
    }

    pub fn pointer(referent: TypeId) -> Self {
        Type {
            kind: TypeKind::Pointer(referent),
            size: 8,
            align: 8,
        }
    }

    pub fn slice(elem: TypeId) -> Self {
        Type {
            kind: TypeKind::Slice(elem),
            size: 16, // ptr + len
            align: 8,
        }
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Slice(_) | TypeKind::Array { .. } | TypeKind::Struct { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    I64,
    U8,
    Bool,
    Unit,
    Never,
    Pointer(TypeId),
    Array(TypeId, usize),
    Slice(TypeId),
    Struct(String, Vec<Field>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub type_id: TypeId,
    pub offset: usize,
}

#[derive(Default)]
pub struct TypeContext {
    types: Vec<Type>,
    cache: HashMap<Type, TypeId>,
}

impl TypeContext {
    pub fn new() -> Self {
        let mut ctx = Self {
            types: Vec::new(),
            cache: HashMap::new(),
        };

        assert_eq!(ctx.intern(Type::i64()), TypeId::I64);
        assert_eq!(ctx.intern(Type::u8()), TypeId::U8);
        assert_eq!(ctx.intern(Type::bool()), TypeId::BOOL);
        assert_eq!(ctx.intern(Type::unit()), TypeId::UNIT);
        assert_eq!(ctx.intern(Type::never()), TypeId::NEVER);

        ctx
    }

    fn intern(&mut self, data: Type) -> TypeId {
        if let Some(&id) = self.cache.get(&data) {
            return id;
        }

        let id = TypeId(self.types.len() as u32);
        self.cache.insert(data.clone(), id);
        self.types.push(data);
        id
    }

    pub fn get(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    pub fn mk_pointer(&mut self, referent: TypeId) -> TypeId {
        let data = Type {
            kind: TypeKind::Pointer(referent),
            size: 8,
            align: 8,
        };
        self.intern(data)
    }

    pub fn mk_array(&mut self, elem: TypeId, len: usize) -> TypeId {
        let elem_data = self.get(elem);
        let data = Type {
            kind: TypeKind::Array(elem, len),
            size: elem_data.size * len,
            align: elem_data.align,
        };
        self.intern(data)
    }

    pub fn mk_slice(&mut self, elem: TypeId) -> TypeId {
        let data = Type {
            kind: TypeKind::Slice(elem),
            size: 16,
            align: 8,
        };
        self.intern(data)
    }

    pub fn mk_empty_struct(&mut self) -> TypeId {
        let data = Type {
            kind: TypeKind::Struct(String::new(), vec![]),
            size: 0,
            align: 1,
        };
        let id = TypeId(self.types.len() as u32);
        self.types.push(data);
        id
    }

    pub fn set_struct(
        &mut self,
        id: TypeId,
        name: String,
        fields: Vec<Field>,
        size: usize,
        align: u64,
    ) {
        self.types[id.0 as usize] = Type {
            kind: TypeKind::Struct(name, fields),
            size,
            align,
        };
    }
}
