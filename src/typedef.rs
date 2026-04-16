use std::io;

use crate::{
    hir::{
        ConstDef, ConstVal, ConstValKind, ExternItem, FunctionDef, ItemKind, Program, StaticDef,
        TypeDef, Visibility,
    },
    types::{TypeContext, TypeId, TypeKind},
};

pub fn emit_typedefs(
    prog: &Program,
    types: &TypeContext,
    out: &mut impl io::Write,
) -> io::Result<()> {
    for use_path in &prog.uses {
        writeln!(out, "use {use_path};")?;
    }

    if !prog.uses.is_empty() {
        writeln!(out)?;
    }

    for item in &prog.items {
        if item.vis != Visibility::Public {
            continue;
        }

        match &item.kind {
            ItemKind::Const(def) => emit_const(def, types, out)?,
            ItemKind::Static(def) => emit_static(def, types, out)?,
            ItemKind::Function(def) => emit_function(def, types, out)?,
            ItemKind::TypeDef(def) => emit_typedef(def, types, out)?,
            ItemKind::Extern(ext) => emit_extern(ext, types, out)?,
        }
    }

    Ok(())
}

fn emit_const(def: &ConstDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    write!(out, "pub const {}: ", def.ident)?;
    emit_type(def.init.type_id, types, out)?;
    write!(out, " = ")?;
    emit_const_val(&def.init, types, out)?;
    writeln!(out, ";")
}

fn emit_static(def: &StaticDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    write!(out, "pub static {}: ", def.ident)?;
    emit_type(def.init.type_id, types, out)?;
    write!(out, " = ")?;
    emit_const_val(&def.init, types, out)?;
    writeln!(out, ";")
}

fn emit_function(
    def: &FunctionDef,
    types: &TypeContext,
    out: &mut impl io::Write,
) -> io::Result<()> {
    write!(out, "pub fn {}(", def.ident)?;

    for (i, param) in def.params.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        write!(out, "{}: ", param.name)?;
        emit_type(param.type_id, types, out)?;
    }

    write!(out, ") -> ")?;
    emit_type(def.return_type_id, types, out)?;
    writeln!(out, ";")
}

fn emit_typedef(def: &TypeDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    let ty = types.get(def.type_id);

    write!(out, "pub struct {} ", def.ident)?;

    if let TypeKind::Struct(_, fields) = &ty.kind {
        writeln!(out, "{{")?;
        for field in fields {
            write!(out, "    {}: ", field.name)?;
            emit_type(field.type_id, types, out)?;
            writeln!(out, ",")?;
        }
        write!(out, "}}")?;
    }

    writeln!(out, " // size: {}, align: {}", ty.size, ty.align)
}

fn emit_extern(ext: &ExternItem, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    match ext {
        ExternItem::Static(def) => {
            write!(out, "extern static {}: ", def.name)?;
            emit_type(def.type_id, types, out)?;
            writeln!(out, ";")
        }
        ExternItem::Function(def) => {
            write!(out, "extern fn {}(", def.name)?;

            for (i, param) in def.params.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}: ", param.name)?;
                emit_type(param.type_id, types, out)?;
            }

            write!(out, ") -> ")?;
            emit_type(def.return_type_id, types, out)?;
            writeln!(out, ";")
        }
    }
}

fn emit_type(type_id: TypeId, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    let ty = types.get(type_id);

    match &ty.kind {
        TypeKind::I64 => write!(out, "i64"),
        TypeKind::F64 => write!(out, "f64"),
        TypeKind::U8 => write!(out, "u8"),
        TypeKind::Bool => write!(out, "bool"),
        TypeKind::Unit => write!(out, "()"),
        TypeKind::Never => write!(out, "!"),
        TypeKind::Opaque => write!(out, "opaque"),
        TypeKind::Pointer(inner) => {
            write!(out, "*")?;
            emit_type(*inner, types, out)
        }
        TypeKind::Array(elem, len) => {
            write!(out, "[")?;
            emit_type(*elem, types, out)?;
            write!(out, "; {len}]")
        }
        TypeKind::Slice(elem) => {
            write!(out, "[")?;
            emit_type(*elem, types, out)?;
            write!(out, "]")
        }
        TypeKind::Struct(name, _) => {
            write!(out, "{name}")
        }
    }
}

fn emit_const_val(val: &ConstVal, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    match &val.kind {
        ConstValKind::Integer(n) => write!(out, "{n}"),
        ConstValKind::Float(n) => write!(out, "{n}"),
        ConstValKind::Bool(b) => write!(out, "{b}"),
        ConstValKind::String(s) => write!(out, "\"{}\"", s.escape_default()),
        ConstValKind::CString(s) => write!(out, "c\"{}\"", s.escape_default()),
        ConstValKind::Struct(field_values) => {
            let TypeKind::Struct(name, fields) = &types.get(val.type_id).kind else {
                unreachable!()
            };
            let (name, fields) = (name.clone(), fields.clone());
            write!(out, "{name} {{")?;
            for (i, (field, field_val)) in fields.iter().zip(field_values.iter()).enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}: ", field.name)?;
                emit_const_val(field_val, types, out)?;
            }
            write!(out, "}}")
        }
        ConstValKind::Array(elems) => {
            write!(out, "[")?;
            for (i, e) in elems.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                emit_const_val(e, types, out)?;
            }
            write!(out, "]")
        }
    }
}
