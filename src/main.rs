use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::{env, fs, process};

use saya::codegen::CodeGen;
use saya::lexer::Lexer;
use saya::parser::Parser;
use saya::type_checker::TypeChecker;
use saya::typedef::emit_typedefs;
use saya::types::TypeContext;

#[derive(Default)]
struct Args {
    input: Option<String>,
    output: Option<String>,
    typedef: Option<String>,
    namespace: Option<String>,
    td_paths: HashMap<String, String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut config = Args::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" => config.output = Some(args.next().ok_or("missing argument for '-o'")?),
            "-t" => config.typedef = Some(args.next().ok_or("missing argument for '-t'")?),
            "-N" => config.namespace = Some(args.next().ok_or("missing argument for '-N'")?),
            "-M" => {
                let mapping = args.next().ok_or("missing argument for '-M'")?;
                let (name, path) = mapping
                    .split_once('=')
                    .ok_or("invalid module mapping, expected '-M name=path'")?;
                config.td_paths.insert(name.into(), path.into());
            }
            s if s.starts_with('-') => return Err(format!("unknown option: '{s}'")),
            path => config.input = Some(path.to_string()),
        }
    }

    Ok(config)
}

fn run() -> Result<(), String> {
    let args = parse_args().map_err(|e| format!("error: {e}"))?;
    let (input, code) = match &args.input {
        Some(path) => (
            path.as_str(),
            fs::read_to_string(path).map_err(|e| format!("error: cannot read `{path}`: {e}"))?,
        ),
        None => {
            let mut code = String::new();
            io::stdin()
                .read_to_string(&mut code)
                .map_err(|e| format!("error: cannot read stdin: {e}"))?;
            ("<stdin>", code)
        }
    };

    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer).map_err(|e| format!("{input}:{e}"))?;
    let program = parser.parse().map_err(|e| format!("{input}:{e}"))?;

    let mut types = TypeContext::new();

    let mut type_checker = TypeChecker::new(&mut types, args.namespace, args.td_paths);
    let typed_program = type_checker
        .check(&program)
        .map_err(|e| format!("{input}:{e}"))?;

    if let Some(td_path) = &args.typedef {
        let mut file = fs::File::create(td_path)
            .map_err(|e| format!("error: cannot create `{td_path}`: {e}"))?;
        emit_typedefs(&typed_program, &types, &mut file)
            .map_err(|e| format!("error: cannot write `{td_path}`: {e}"))?;
    }

    let mut code_gen = CodeGen::new(&mut types);
    let qbe_il = code_gen
        .generate(&typed_program)
        .map_err(|e| format!("{input}:{e}"))?;

    match &args.output {
        Some(path) => {
            fs::write(path, qbe_il).map_err(|e| format!("error: cannot write `{path}`: {e}"))?
        }
        None => io::stdout()
            .write_all(qbe_il.as_bytes())
            .map_err(|e| format!("error: cannot write stdout: {e}"))?,
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        process::exit(1);
    }
}
