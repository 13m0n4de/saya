use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TypeId {
    U8,
    U16,
    U32,
    U64,

    I8,
    I16,
    I32,
    I64,

    F32,
    F64,

    Bool,

    Unit,
    Never,
    Opaque,

    Interned(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub kind: TypeKind,
    pub size: usize,
    pub align: usize,
}

impl Type {
    pub const U8: Self = Type {
        kind: TypeKind::U8,
        size: 1,
        align: 1,
    };
    pub const U16: Self = Type {
        kind: TypeKind::U16,
        size: 2,
        align: 2,
    };
    pub const U32: Self = Type {
        kind: TypeKind::U32,
        size: 4,
        align: 4,
    };
    pub const U64: Self = Type {
        kind: TypeKind::U64,
        size: 8,
        align: 8,
    };

    pub const I8: Self = Type {
        kind: TypeKind::I8,
        size: 1,
        align: 1,
    };
    pub const I16: Self = Type {
        kind: TypeKind::I16,
        size: 2,
        align: 2,
    };
    pub const I32: Self = Type {
        kind: TypeKind::I32,
        size: 4,
        align: 4,
    };
    pub const I64: Self = Type {
        kind: TypeKind::I64,
        size: 8,
        align: 8,
    };

    pub const F32: Self = Type {
        kind: TypeKind::F32,
        size: 4,
        align: 4,
    };
    pub const F64: Self = Type {
        kind: TypeKind::F64,
        size: 8,
        align: 8,
    };

    pub const BOOL: Self = Type {
        kind: TypeKind::Bool,
        size: 1,
        align: 1,
    };
    pub const UNIT: Self = Type {
        kind: TypeKind::Unit,
        size: 0,
        align: 1,
    };
    pub const NEVER: Self = Type {
        kind: TypeKind::Never,
        size: 0,
        align: 1,
    };
    pub const OPAQUE: Self = Type {
        kind: TypeKind::Opaque,
        size: 0,
        align: 1,
    };
}

impl Type {
    pub fn pointer(referent: TypeId) -> Self {
        Type {
            kind: TypeKind::Pointer(referent),
            size: 8,
            align: 8,
        }
    }

    pub fn array(elem: TypeId, len: usize, elem_size: usize, elem_align: usize) -> Self {
        Type {
            kind: TypeKind::Array(elem, len),
            size: elem_size * len,
            align: elem_align,
        }
    }

    pub fn slice(elem: TypeId) -> Self {
        Type {
            kind: TypeKind::Slice(elem),
            size: 16, // ptr + len
            align: 8,
        }
    }

    pub fn func(params_type: Vec<TypeId>, return_type: TypeId, is_variadic: bool) -> Self {
        Type {
            kind: TypeKind::Fn(params_type, return_type, is_variadic),
            size: 8,
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
    U8,
    U16,
    U32,
    U64,

    I8,
    I16,
    I32,
    I64,

    F32,
    F64,

    Bool,

    Unit,
    Never,
    Opaque,
    Pointer(TypeId),

    Array(TypeId, usize),
    Slice(TypeId),
    Struct(String, Vec<Field>),
    Fn(Vec<TypeId>, TypeId, bool),
}

impl TypeKind {
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            TypeKind::U8 | TypeKind::U16 | TypeKind::U32 | TypeKind::I32 | TypeKind::I64
        )
    }
    pub fn is_float(&self) -> bool {
        matches!(self, TypeKind::F32 | TypeKind::F64)
    }
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub type_id: TypeId,
    pub offset: usize,
}

#[derive(Default)]
pub struct TypeContext {
    interned: Vec<Type>,
    cache: HashMap<Type, TypeId>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            interned: Vec::new(),
            cache: HashMap::new(),
        }
    }

    fn intern(&mut self, data: Type) -> TypeId {
        if let Some(&id) = self.cache.get(&data) {
            return id;
        }

        let id = TypeId::Interned(self.interned.len() as u32);
        self.cache.insert(data.clone(), id);
        self.interned.push(data);
        id
    }

    pub fn get(&self, id: TypeId) -> &Type {
        match id {
            TypeId::U8 => &Type::U8,
            TypeId::U16 => &Type::U16,
            TypeId::U32 => &Type::U32,
            TypeId::U64 => &Type::U64,
            TypeId::I8 => &Type::I8,
            TypeId::I16 => &Type::I16,
            TypeId::I32 => &Type::I32,
            TypeId::I64 => &Type::I64,
            TypeId::F32 => &Type::F32,
            TypeId::F64 => &Type::F64,
            TypeId::Bool => &Type::BOOL,
            TypeId::Unit => &Type::UNIT,
            TypeId::Never => &Type::NEVER,
            TypeId::Opaque => &Type::OPAQUE,
            TypeId::Interned(n) => &self.interned[n as usize],
        }
    }

    pub fn mk_pointer(&mut self, referent: TypeId) -> TypeId {
        self.intern(Type::pointer(referent))
    }

    pub fn mk_array(&mut self, elem: TypeId, len: usize) -> TypeId {
        let elem_data = self.get(elem);
        self.intern(Type::array(elem, len, elem_data.size, elem_data.align))
    }

    pub fn mk_slice(&mut self, elem: TypeId) -> TypeId {
        self.intern(Type::slice(elem))
    }

    pub fn mk_empty_struct(&mut self) -> TypeId {
        let data = Type {
            kind: TypeKind::Struct(String::new(), vec![]),
            size: 0,
            align: 1,
        };
        let id = TypeId::Interned(self.interned.len() as u32);
        self.interned.push(data);
        id
    }

    pub fn set_struct(
        &mut self,
        id: TypeId,
        name: String,
        fields: Vec<Field>,
        size: usize,
        align: usize,
    ) {
        let TypeId::Interned(n) = id else {
            unreachable!()
        };
        self.interned[n as usize] = Type {
            kind: TypeKind::Struct(name, fields),
            size,
            align,
        };
    }

    pub fn mk_fn(
        &mut self,
        params_type: Vec<TypeId>,
        return_type: TypeId,
        is_variadic: bool,
    ) -> TypeId {
        self.intern(Type::func(params_type, return_type, is_variadic))
    }

    pub fn type_name(&self, id: TypeId) -> String {
        match &self.get(id).kind {
            TypeKind::U8 => "u8".into(),
            TypeKind::U16 => "u16".into(),
            TypeKind::U32 => "u32".into(),
            TypeKind::U64 => "u64".into(),
            TypeKind::I8 => "i8".into(),
            TypeKind::I16 => "i16".into(),
            TypeKind::I32 => "i32".into(),
            TypeKind::I64 => "i64".into(),
            TypeKind::F32 => "f32".into(),
            TypeKind::F64 => "f64".into(),
            TypeKind::Bool => "bool".into(),
            TypeKind::Unit => "()".into(),
            TypeKind::Never => "!".into(),
            TypeKind::Opaque => "opaque".into(),
            TypeKind::Pointer(inner) => format!("*{}", self.type_name(*inner)),
            TypeKind::Array(elem, len) => format!("[{}; {len}]", self.type_name(*elem)),
            TypeKind::Slice(elem) => format!("[{}]", self.type_name(*elem)),
            TypeKind::Struct(name, _) => name.into(),
            TypeKind::Fn(params_type, return_type, is_variadic) => {
                let mut parts: Vec<String> =
                    params_type.iter().map(|id| self.type_name(*id)).collect();
                if *is_variadic {
                    parts.push("...".to_string());
                }
                let ret = if *return_type == TypeId::Unit {
                    String::new()
                } else {
                    format!(" -> {}", self.type_name(*return_type))
                };
                format!("fn({}){}", parts.join(", "), ret)
            }
        }
    }

    pub fn is_assignable(&self, from: TypeId, to: TypeId) -> bool {
        // from any pointer to *opaque
        if matches!(
            self.get(to).kind,
            TypeKind::Pointer(inner) if matches!(self.get(inner).kind, TypeKind::Opaque)
        ) {
            return matches!(self.get(from).kind, TypeKind::Pointer(_));
        }

        from == to
    }
}
