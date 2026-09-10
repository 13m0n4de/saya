use std::io;

use crate::{
    hir::{
        ConstDef, ConstVal, ConstValKind, ExternItem, FunctionDef, ItemKind, Param, Program,
        StaticDef, TypeAlias, TypeDef, Visibility,
    },
    types::{TypeContext, TypeKind},
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
            ItemKind::TypeAlias(def) => emit_type_alias(def, types, out)?,
            ItemKind::Extern(ext) => emit_extern(ext, types, out)?,
        }
    }

    Ok(())
}

fn emit_const(def: &ConstDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    write!(
        out,
        "pub const {}: {} = ",
        def.ident,
        types.type_name(def.init.type_id)
    )?;
    emit_const_val(&def.init, types, out)?;
    writeln!(out, ";")
}

fn emit_static(def: &StaticDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    write!(
        out,
        "pub static {}: {} = ",
        def.ident,
        types.type_name(def.init.type_id)
    )?;
    emit_const_val(&def.init, types, out)?;
    writeln!(out, ";")
}

fn emit_params(params: &[Param], types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        write!(out, "{}: {}", param.name, types.type_name(param.type_id))?;
    }
    Ok(())
}

fn emit_function(
    def: &FunctionDef,
    types: &TypeContext,
    out: &mut impl io::Write,
) -> io::Result<()> {
    write!(out, "pub fn {}(", def.ident)?;
    emit_params(&def.params, types, out)?;
    write!(out, ") -> ")?;
    writeln!(out, "{};", types.type_name(def.return_type_id))
}

fn emit_typedef(def: &TypeDef, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    let ty = types.get(def.type_id);

    write!(out, "pub struct {} ", def.ident)?;

    if let TypeKind::Struct(_, fields) = &ty.kind {
        writeln!(out, "{{")?;
        for field in fields {
            writeln!(
                out,
                "    {}: {},",
                field.name,
                types.type_name(field.type_id)
            )?;
        }
        write!(out, "}}")?;
    }

    writeln!(out, " // size: {}, align: {}", ty.size, ty.align)
}

fn emit_type_alias(
    def: &TypeAlias,
    types: &TypeContext,
    out: &mut impl io::Write,
) -> io::Result<()> {
    writeln!(
        out,
        "pub type {} = {};",
        def.ident,
        types.type_name(def.type_id)
    )
}

fn emit_extern(ext: &ExternItem, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    match ext {
        ExternItem::Static(def) => {
            writeln!(
                out,
                "@symbol(\"{}\") extern static {}: {};",
                def.symbol,
                def.ident,
                types.type_name(def.type_id)
            )
        }
        ExternItem::Function(def) => {
            write!(out, "@symbol(\"{}\") extern fn {}(", def.symbol, def.ident)?;
            emit_params(&def.params, types, out)?;
            if def.is_variadic {
                write!(out, ", ...")?;
            }
            writeln!(out, ") -> {};", types.type_name(def.return_type_id))
        }
    }
}

fn emit_const_val(val: &ConstVal, types: &TypeContext, out: &mut impl io::Write) -> io::Result<()> {
    match &val.kind {
        ConstValKind::Integer(n) => write!(out, "{n}{}", types.type_name(val.type_id)),
        ConstValKind::Float(n) => write!(out, "{n}{}", types.type_name(val.type_id)),
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
        ConstValKind::Repeat(elem, count) => {
            write!(out, "[")?;
            emit_const_val(elem, types, out)?;
            write!(out, "; {count}]")
        }
    }
}
