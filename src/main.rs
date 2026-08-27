use oxide_ai_pssa::cli::CLIHandler;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    CLIHandler::parse_and_execute(args);
}
