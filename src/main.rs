use std::collections::HashMap;
use std::{env, error::Error, fs, process};

use saya::codegen::CodeGen;
use saya::lexer::Lexer;
use saya::parser::Parser;
use saya::type_checker::TypeChecker;
use saya::typedef::emit_typedefs;
use saya::types::TypeContext;

struct Args {
    input: String,
    output: String,
    typedef: Option<String>,
    namespace: Option<String>,
    td_paths: HashMap<String, String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut config = Args {
        input: String::new(),
        output: "out.ssa".to_string(),
        typedef: None,
        namespace: None,
        td_paths: HashMap::new(),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" => config.output = args.next().ok_or("missing argument for '-o'")?,
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
            path => config.input = path.to_string(),
        }
    }

    if config.input.is_empty() {
        return Err("no input file".to_string());
    }

    Ok(config)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    let code = fs::read_to_string(&args.input)?;

    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer)?;
    let program = parser.parse()?;

    let mut types = TypeContext::new();

    let mut type_checker = TypeChecker::new(&mut types, args.namespace, args.td_paths);
    let typed_program = type_checker.check_program(&program)?;

    if let Some(td_path) = &args.typedef {
        let mut file = fs::File::create(td_path)?;
        emit_typedefs(&typed_program, &types, &mut file)?;
    }

    let mut code_gen = CodeGen::new(&mut types);
    let qbe_il = code_gen.generate(&typed_program)?;

    fs::write(args.output, qbe_il)?;

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
