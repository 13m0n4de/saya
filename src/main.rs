use std::{env, error::Error, fs::read_to_string, process};

use saya::codegen::CodeGen;
use saya::lexer::Lexer;
use saya::parser::Parser;
use saya::type_checker::TypeChecker;
use saya::types::TypeContext;

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return Err("no saya file provided".into());
    }

    let input_file = &args[1];

    let code = read_to_string(input_file)?;

    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer)?;
    let program = parser.parse()?;

    let mut types = TypeContext::new();

    let mut type_checker = TypeChecker::new(&mut types);
    let typed_program = type_checker.check_program(&program)?;

    let mut code_gen = CodeGen::new(&mut types);
    let qbe_il = code_gen.generate(&typed_program)?;

    std::fs::write("out.ssa", qbe_il)?;

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
