#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub kind: TypeKind,
    pub size: usize,
    pub align: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    I64,
    U8,
    Bool,
    Pointer(Box<Type>),
    Array(Box<Type>, usize),
    Slice(Box<Type>),
    Struct(StructType),
    Unit,
    Never,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub name: String,
    pub fields: Vec<FieldInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInfo {
    pub name: String,
    pub ty: Type,
    pub offset: usize,
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

    pub fn pointer(referent: Type) -> Self {
        Type {
            kind: TypeKind::Pointer(Box::new(referent)),
            size: 8,
            align: 8,
        }
    }

    pub fn array(elem: Type, count: usize) -> Self {
        let size = elem.size * count;
        let align = elem.align;
        Type {
            kind: TypeKind::Array(Box::new(elem), count),
            size,
            align,
        }
    }

    pub fn slice(elem: Type) -> Self {
        Type {
            kind: TypeKind::Slice(Box::new(elem)),
            size: 16, // ptr + len
            align: 8,
        }
    }

    pub fn struct_type(name: String, fields: Vec<FieldInfo>, size: usize, align: u64) -> Self {
        Type {
            kind: TypeKind::Struct(StructType { name, fields }),
            size,
            align,
        }
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Slice(..) | TypeKind::Array(..) | TypeKind::Struct(..)
        )
    }

    pub fn to_qbe_base(&self) -> qbe::Type<'static> {
        match self.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::Word,
            TypeKind::Bool => qbe::Type::Word,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Array(_, _) => qbe::Type::Long,
            TypeKind::Slice(_) => qbe::Type::Long,
            TypeKind::Struct(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
        }
    }

    pub fn to_qbe_load(&self) -> qbe::Type<'static> {
        match self.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::UnsignedByte,
            TypeKind::Bool => qbe::Type::UnsignedByte,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Array(_, _) => qbe::Type::Long,
            TypeKind::Slice(_) => qbe::Type::Long,
            TypeKind::Struct(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
        }
    }

    pub fn to_qbe_store(&self) -> qbe::Type<'static> {
        match self.kind {
            TypeKind::I64 => qbe::Type::Long,
            TypeKind::U8 => qbe::Type::Byte,
            TypeKind::Bool => qbe::Type::Byte,
            TypeKind::Pointer(_) => qbe::Type::Long,
            TypeKind::Array(_, _) => qbe::Type::Long,
            TypeKind::Slice(_) => qbe::Type::Long,
            TypeKind::Struct(_) => qbe::Type::Long,
            TypeKind::Unit => qbe::Type::Long,
            TypeKind::Never => qbe::Type::Long,
        }
    }
}
