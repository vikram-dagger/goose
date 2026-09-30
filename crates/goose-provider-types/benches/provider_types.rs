//! Benchmarks for the provider-agnostic hot paths that run on every agent turn:
//! conversation repair, wire-format conversion, tool-argument parsing, and
//! canonical model lookups.

use divan::{black_box, Bencher};
use goose_provider_types::canonical::maybe_get_canonical_model;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::{fix_conversation, Conversation};
use goose_provider_types::formats::{anthropic, openai};
use goose_provider_types::images::ImageFormat;
use goose_provider_types::json::{parse_tool_arguments, safely_parse_json};
use goose_provider_types::utils::sanitize_unicode_tags;
use rmcp::model::{object, CallToolRequestParams, CallToolResult, ContentBlock};
use serde_json::json;

fn main() {
    divan::main();
}

const TURN_COUNTS: &[usize] = &[10, 100];

/// Build a realistic conversation made of user prompts, assistant tool calls,
/// tool responses, and assistant replies.
fn build_messages(turns: usize) -> Vec<Message> {
    let mut messages = Vec::with_capacity(turns * 4);
    for i in 0..turns {
        let id = format!("call_{i}");
        messages.push(Message::user().with_text(format!(
            "Please read src/module_{i}.rs and summarize what the functions do."
        )));
        messages.push(
            Message::assistant()
                .with_text("Let me read that file for you.")
                .with_tool_request(
                    id.clone(),
                    Ok(
                        CallToolRequestParams::new("developer__text_editor").with_arguments(
                            object(json!({
                                "command": "view",
                                "path": format!("/workspace/project/src/module_{i}.rs"),
                            })),
                        ),
                    ),
                ),
        );
        messages.push(Message::user().with_tool_response(
            id,
            Ok(CallToolResult::success(vec![ContentBlock::text(
                "fn main() {\n    println!(\"hello\");\n}\n".repeat(20),
            )])),
        ));
        messages.push(Message::assistant().with_text(format!(
            "The module {i} defines a `main` function that prints a greeting."
        )));
    }
    messages
}

mod conversation {
    use super::*;

    #[divan::bench(args = TURN_COUNTS)]
    fn fix(bencher: Bencher, turns: usize) {
        let messages = build_messages(turns);
        bencher
            .with_inputs(|| Conversation::new_unvalidated(messages.clone()))
            .bench_values(|conversation| black_box(fix_conversation(conversation)));
    }

    #[divan::bench(args = TURN_COUNTS)]
    fn agent_visible_messages(bencher: Bencher, turns: usize) {
        let conversation = Conversation::new_unvalidated(build_messages(turns));
        bencher.bench(|| black_box(conversation.agent_visible_messages()));
    }
}

mod formats {
    use super::*;

    #[divan::bench(args = TURN_COUNTS)]
    fn openai_format_messages(bencher: Bencher, turns: usize) {
        let messages = build_messages(turns);
        bencher.bench(|| black_box(openai::format_messages(&messages, &ImageFormat::OpenAi)));
    }

    #[divan::bench(args = TURN_COUNTS)]
    fn anthropic_format_messages(bencher: Bencher, turns: usize) {
        let messages = build_messages(turns);
        bencher.bench(|| black_box(anthropic::format_messages(&messages)));
    }
}

mod json {
    use super::*;

    const VALID_ARGS: &str = r#"{"command":"write","path":"/workspace/project/src/lib.rs","file_text":"pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n","options":{"create_dirs":true,"mode":"0644"}}"#;
    const CONTROL_CHAR_ARGS: &str = "{\"command\":\"write\",\"path\":\"/workspace/project/src/lib.rs\",\"file_text\":\"pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\"}";
    const DOUBLE_ENCODED_ARGS: &str =
        r#""{\"command\":\"view\",\"path\":\"/workspace/project/README.md\"}""#;

    #[divan::bench]
    fn parse_tool_arguments_valid() -> Option<serde_json::Value> {
        parse_tool_arguments(black_box(VALID_ARGS))
    }

    #[divan::bench]
    fn parse_tool_arguments_double_encoded() -> Option<serde_json::Value> {
        parse_tool_arguments(black_box(DOUBLE_ENCODED_ARGS))
    }

    #[divan::bench]
    fn safely_parse_json_control_chars() -> Result<serde_json::Value, serde_json::Error> {
        safely_parse_json(black_box(CONTROL_CHAR_ARGS))
    }
}

mod text {
    use super::*;

    #[divan::bench]
    fn sanitize_unicode_tags_4kb(bencher: Bencher) {
        let text = "Hello, world! Caf\u{0065}\u{0301} \u{E0041}\u{E0042} ".repeat(128);
        bencher.bench(|| black_box(sanitize_unicode_tags(black_box(&text))));
    }
}

mod canonical {
    use super::*;

    #[divan::bench(args = [
        ("anthropic", "claude-sonnet-4-5"),
        ("openai", "gpt-4o"),
        ("openrouter", "anthropic/claude-3.5-sonnet"),
    ])]
    fn lookup(bencher: Bencher, args: (&str, &str)) {
        // Load the bundled registry up front so the one-time lazy initialization
        // is not attributed to the lookup itself.
        let _ = maybe_get_canonical_model(args.0, args.1);
        bencher.bench(|| maybe_get_canonical_model(black_box(args.0), black_box(args.1)));
    }
}
