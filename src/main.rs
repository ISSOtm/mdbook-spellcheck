#![doc(html_root_url = "https://docs.rs/mdbook-spellcheck/0.1.0")]

use anyhow::Context;
use codespan_reporting::diagnostic::{Diagnostic, Label, Severity};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::{self, Config};
use lazy_static::lazy_static;
use mdbook::book::BookItem;
use mdbook::config::BookConfig;
use mdbook::errors::{Error, Result};
use mdbook::renderer::{RenderContext, Renderer};
use nlprule::types::Suggestion;
use nlprule::{rules_filename, tokenizer_filename, Rules, Tokenizer};
use pulldown_cmark::{CodeBlockKind, Event, OffsetIter, Parser, Tag};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io::{self, Write};
use std::ops::{Add, Range};
use std::path::Path;
use std::str::FromStr;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use toml::value::Table;

/// Outputs a diagnostic with its "type" bold & colored, then prints its message (with [`eprintln`]).
///
/// Example:
/// ```rust
/// use termcolor::{Color, ColorSpec, StandardStream};
///
/// # fn main() {
/// diagnostic!(Color::Yellow, "warning:", " Found the answer, it was {}", 42);
/// # }
/// ```
macro_rules! diagnostic {
    ($color:expr, $diag_type:expr, $($msg:tt)+) => {
        let mut stderr = StandardStream::stderr(auto_color(atty::Stream::Stderr));

        stderr.set_color(ColorSpec::new().set_fg(Some($color)).set_bold(true)).unwrap();
        write!(&mut stderr, $diag_type).unwrap();

        stderr.reset().unwrap();
        eprintln!($($msg)+);
    };
}

#[doc(hidden)]
fn main() -> Result<()> {
    let mut stdin = io::stdin();
    let ctx = RenderContext::from_json(&mut stdin)?;

    let renderer = SpellChecker;

    if ctx.version != mdbook::MDBOOK_VERSION {
        // We should probably use the `semver` crate to check compatibility
        // here...
        diagnostic!(
            Color::Yellow,
            "warning:",
            " The {} renderer was built against version {} of mdbook, \
             but we're being called from version {}",
            renderer.name(),
            mdbook::MDBOOK_VERSION,
            ctx.version,
        );
    }

    renderer.render(&ctx)
}

/// Default language to use for spell checking.
/// Debug builds check that this is equal to `mdbook`'s default language.
///
/// This is only separate so that this lives longer than a temp copy created from a default book config would.
static DEFAULT_LANG: &str = "en";

/// The spell-checking "renderer".
struct SpellChecker;

impl Renderer for SpellChecker {
    fn name(&self) -> &'static str {
        "spellcheck"
    }

    fn render(&self, ctx: &RenderContext) -> Result<()> {
        debug_assert_eq!(DEFAULT_LANG, &BookConfig::default().language.unwrap());

        // Use the book's main language
        let lang = ctx
            .config
            .book
            .language
            .as_ref()
            .map_or(DEFAULT_LANG, |lang| lang);
        // Warn about the lack of multilingual handling
        if ctx.config.book.multilingual {
            diagnostic!(
                Color::Yellow,
                "warning:",
                " Spell checking will only be performed against the book's main language!",
            );
        }

        // Get the rule configuration
        let rules_config = RulesConfig::new(
            ctx.config
                .get_renderer(self.name())
                .and_then(|table| table.get("rules"))
                .map(|value| {
                    value.as_table().ok_or_else(|| {
                        Error::msg(format!(
                            "`output.{}.rules` in config should be a table, not {}",
                            self.name(),
                            value.type_str()
                        ))
                    })
                })
                .transpose()?,
        )
        .context("Bad spellcheck rules config")?;

        let (tokenizer, rules) = SpellChecker::load_checker(&lang, &rules_config)?;
        /// Run some code on the suggestions generated from a string, if any are.
        /// This is a macro so that inner code is allowed to `?` out of `render` and not a closure
        macro_rules! if_suggestions {
            ($string:expr, |$suggestions:ident| $code:stmt) => {
                let $suggestions = rules.suggest($string, &tokenizer);
                if !$suggestions.is_empty() {
                    $code
                }
            };
        }

        // Set up codespan_reporting's output
        let mut output = StandardStream::stderr(auto_color(atty::Stream::Stderr));
        let config = term::Config::default();

        let mut scribe = SuggestionScribe::new(&mut output, &config, &rules_config);

        // Set up SUMMARY.md book structure iterator
        let mut summary_path = ctx.root.join(&ctx.config.book.src);
        summary_path.push("SUMMARY.md");
        let summary = fs::read_to_string(&summary_path)
            .context(format!("Failed to read {}", summary_path.display()))?;

        let summary_file = SimpleFile::new("SUMMARY.md", &summary);
        let summary_locs = parse_summary(&summary, |events: &mut _| {
            let mut name = String::new();
            let mut mappings = Vec::new();
            let ending_tag =
                get_block_text(events, &mut name, &mut mappings, lang, |title, span| {
                    if_suggestions!(&title, |suggestions| {
                        // `span` points inside the link/image, but the
                        // suggestion's span points inside the title!
                        // Correct this.
                        let span = title_span(span, &summary);
                        for suggestion in suggestions {
                            scribe.report(
                                &suggestion,
                                &summary_file,
                                shift(suggestion.span().byte(), span.start),
                            );
                        }
                    });
                });
            ending_tag.map(|tag| (name, mappings, tag))
        })
        .context("Failed to parse summary")?;
        let mut summary_locs = summary_locs.iter();

        //
        // Setup finished, actual checking begins here
        //

        let mut cur_part = Part::new(PartType::Prefix);

        for item in ctx.book.iter() {
            // Do not fetch the next item if it doesn't have any content
            if matches!(item, BookItem::Separator) {
                continue;
            }

            match item {
                BookItem::Chapter(chapter) => {
                    let mappings = loop {
                        match summary_locs.next() {
                            // mdBook does not notify when we begin suffix chapters, so check here
                            Some(SummaryItem::SuffixPart) => cur_part = Part::new(PartType::Suffix),
                            // The titles may not match, for example if the part name contains
                            // inline code (due to the placeholder), so we don't check here
                            Some(SummaryItem::Chapter(_, mappings)) => break mappings,
                            item => {
                                panic!(
                                    "mdBook parsed chapter \"{}\", but we parsed {:?}",
                                    chapter.name, item
                                )
                            }
                        }
                    };

                    let mut first_chapter_err = true;
                    let mut report_chapter = || {
                        if first_chapter_err {
                            first_chapter_err = false;
                            cur_part.report();

                            let mut stderr =
                                StandardStream::stderr(auto_color(atty::Stream::Stderr));
                            stderr.set_color(ColorSpec::new().set_bold(true)).unwrap();
                            writeln!(&mut stderr, "In chapter \"{}\":", &chapter.name).unwrap();
                            stderr.reset().unwrap();
                            eprintln!();
                        }
                    };

                    if_suggestions!(&chapter.name, |suggestions| {
                        report_chapter();
                        for suggestion in suggestions {
                            scribe.report(
                                &suggestion,
                                &summary_file,
                                find_orig_span(suggestion.span().byte(), &mappings).context(
                                    format!(
                                        "Failed to find {:?} in \"{}\"'s mappings: {:#?}",
                                        suggestion, chapter.name, mappings
                                    ),
                                )?,
                            );
                        }
                    });

                    let src_path = match &chapter.source_path {
                        Some(path) => path,
                        // If the chapter has no file, it's a draft, and thus has no contents.
                        // We can safely skip those.
                        None => continue,
                    };
                    // We must `format!()` because `std::path::Display` is `Display` but not `Clone`
                    let content =
                        SimpleFile::new(format!("{}", src_path.display()), &chapter.content);

                    // For each "block", build a representation of its text, and spell-check that
                    let mut txt = String::new();
                    let mut offsets = Vec::new();
                    let mut events = Parser::new(&chapter.content).into_offset_iter().peekable();
                    while let Some((evt, span)) = events.next() {
                        match evt {
                            Event::Start(Tag::List(_)) => (), // Ignore lists, we care about the items within
                            Event::End(Tag::List(_)) => (), // Consequently, ignore their endings as well

                            Event::Start(tag) if !is_inline(&tag) => {
                                // List items may contain a paragraph if they are "loose", i.e.
                                // they are separated by blank lines; this generates a paragraph
                                // inside the list item.
                                // FIXME: List items can actually contain blocks, so we need an entire refactor
                                let contains_para =matches!(events.peek(),Some((Event::Start(Tag::Paragraph), _)));
                                if contains_para {
                                    events.next(); // Skip that paragraph start
                                }

                                txt.clear();
                                offsets.clear();

                                let ending_tag =
                                    get_block_text(
                                        &mut events, &mut txt, &mut offsets, lang, |title, span| {
                                            if_suggestions!(&title, |suggestions| {
                                                report_chapter();
                                                // `span` points inside the link/image, but the
                                                // suggestion's span points inside the title!
                                                // Correct this.
                                                let span = title_span(span, &chapter.content);
                                                for suggestion in suggestions {
                                                    scribe.report(
                                                        &suggestion,
                                                        &content,
                                                        shift(suggestion.span().byte(), span.start),
                                                    );
                                                }
                                            });
                                        })
                                    .context(format!(
                                        "Error within block {:?} (\"{}\")",
                                        tag, &chapter.content[span]))
                                    .context(format!("Failed to parse chapter \"{}\"", chapter.name))?;
                                if contains_para {
                                    assert_eq!(ending_tag, Tag::Paragraph);
                                    assert_eq!(events.next().map(|t|t.0), Some(Event::End(tag.clone())));
                                } else {
                                    assert_eq!(ending_tag, tag);
                                }

                                match tag {
                                    // Do not spell check code block contents, unless explicitly
                                    // requested by an attribute (`rust,spellcheck` for example)
                                    Tag::CodeBlock(CodeBlockKind::Fenced(lang))
                                        if !lang.split(',').any(|elem| {
                                            elem.trim().eq_ignore_ascii_case("spellcheck")
                                        }) => {}

                                    _ => {
                                        // Spell check the block's contents
                                        if_suggestions!(&txt, |suggestions| {
                                            report_chapter();
                                            for suggestion in suggestions {
                                                // Span within the "cleaned-up" string
                                                let str_span = suggestion.span().byte();
                                                // ... but we want a span within the original string!
                                                let range = find_orig_span(str_span, &offsets)
                                                    .context(format!(
                                                        "Failed to find {:?} in \"{}\"'s mappings: {:#?}",
                                                        suggestion, txt, offsets
                                                    ))?;

                                                scribe.report(&suggestion, &content, range);
                                            }
                                        });

                                    }
                                }
                            }

                            Event::Html(_) => (), // TODO: it would be nice to try extracting content and spell checking that

                            Event::Start(tag) => {
                                return Err(Error::msg(format!(
                                    "Unexpected inline {:?} outside of a block (in chapter {})",
                                    tag, chapter.name
                                )))
                            }
                            // Given that we're filtering all block starts, we shouldn't see any endings
                            // The endings to the processed blocks are handled by the starting event's
                            // code, lists are explicitly ignored (above), the rest of starts
                            // produces an error (just above); so if we see anything, it would be
                            // `pulldown_cmark` screwing up.
                            Event::End(_) => unreachable!(),

                            Event::Text(_) => return Err(Error::msg(format!("Unexpected text outside of any blocks (in chapter {})", chapter.name))),
                            Event::Code(_) => return Err(Error::msg(format!("Unexpected code outside of any blocks (in chapter {})", chapter.name))),
                            Event::FootnoteReference(_) => return Err(Error::msg(format!("Unexpected footnote reference outside of any blocks (in chapter {})", chapter.name))),
                            Event::TaskListMarker(_) => return Err(Error::msg(format!("Unexpected task list marker outside of any blocks (in chapter {})", chapter.name))),

                            Event::SoftBreak | Event::HardBreak | Event::Rule => (), // We can just ignore those
                        }
                    }
                }

                BookItem::PartTitle(title) => {
                    let mappings = match summary_locs.next() {
                        // The titles may not match, for example if the part name contains inline
                        // code (due to the placeholder), so don't check if the titles match
                        Some(SummaryItem::Part(name, mappings)) if name == title => mappings,
                        item => {
                            panic!("mdBook parsed part \"{}\", but we parsed {:?}", title, item)
                        }
                    };

                    cur_part = Part::new(PartType::Part(&title));

                    if_suggestions!(&title, |suggestions| {
                        cur_part.report();
                        eprintln!();
                        for suggestion in suggestions {
                            scribe.report(
                                &suggestion,
                                &summary_file,
                                find_orig_span(suggestion.span().byte(), &mappings).context(
                                    format!(
                                        "Failed to find {:?} in \"{}\"'s mappings: {:#?}",
                                        suggestion, title, mappings
                                    ),
                                )?,
                            );
                        }
                    });
                }

                BookItem::Separator => unreachable!(),
            }
        }

        // TODO: keep track of errors separately, and error out if any
        if scribe.nb_reports() != 0 {
            eprintln!("Found {} spell checking reports", scribe.nb_reports());
        }

        Ok(())
    }
}

/// What [the CommonMark spec](https://spec.commonmark.org/0.29/#whitespace-character) calls "whitespace":
///
/// > A whitespace character is a space (U+0020), tab (U+0009), newline (U+000A), line tabulation
/// > (U+000B), form feed (U+000C), or carriage return (U+000D).
static COMMONMARK_WHITESPACE: &[char] = &[' ', '\t', '\n', '\t', '\u{000C}', '\r'];

/// Takes a span over a link, and produces a span over only that link's title, assuming it has one.
fn title_span(mut span: Range<usize>, text: &str) -> Range<usize> {
    // It's actually much easier to start from the end, and work our way backwards!
    let mut iter = text[span.clone()].char_indices().rev();

    assert_eq!(iter.next().map(|(_, c)| c), Some(')'));
    let mut iter = iter.skip_while(|(_, c)| COMMONMARK_WHITESPACE.contains(c));
    let (end_ofs, end_c) = iter
        .next()
        .expect("Nothing but whitespace and a closing paren??");
    // The link title ends at this character (not inclusive, and that's correct)
    span.end = span.start + end_ofs;
    let opening_char = match end_c {
        '"' => '"',
        '\'' => '\'',
        ')' => '(',
        c => panic!("Unexpected title closing character '{}'", c),
    };

    // Work our way to the unescaped, uh, opening character
    let ofs = loop {
        let (ofs, c) = iter.next().expect("Unexpected unopened title");
        // If we found the opening character, and it's not escaped, we're at the end
        // It's OK to consume the char before it, because there may only be one unescaped opening char
        if c == opening_char
            && iter
                .next()
                .expect("Expected something before title opening")
                .1
                != '\\'
        {
            break ofs;
        }
    };

    debug_assert_eq!(opening_char.len_utf8(), 1);
    span.start += ofs + 1;
    span
}

/// Computes a text-only representation of a markdown block, keeping track of the mappings as it goes.
/// The tag returned is the ending block's.
fn get_block_text<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>>(
    events: &mut I,
    text: &mut String,
    mappings: &mut Vec<StringMapping>,
    lang: &str,
    mut title_callback: impl FnMut(&str, Range<usize>),
) -> Result<Tag<'a>> {
    lazy_static! {
        /// Localized inline code tag placeholder.
        static ref CODE_PLACEHOLDERS: HashMap<&'static str,&'static str> = [("en","<code>"),("fr","<code>")].iter().cloned().collect();
    }

    let mut depth = 0;

    loop {
        let (evt, span) = events.next().expect("Unterminated block??");
        match evt {
            Event::Start(tag) if is_inline(&tag) => {
                // Spell check link and image titles
                if let Tag::Link(_, _, title) | Tag::Image(_, _, title) = tag {
                    title_callback(&title, span);
                }
                depth += 1; // Process inline markup
            }
            Event::Start(_) => {
                // This begins a new block, but we're already inside of one!
                return Err(Error::msg(format!(
                    "Already inside a block, but begun a {:?}",
                    evt
                )));
            }

            Event::Text(content) => {
                mappings.push(StringMapping::new(&span, text.len()));
                text.push_str(&content);
            }
            Event::Code(_) => {
                text.push_str(CODE_PLACEHOLDERS.get(lang).unwrap_or_else(|| {
                    panic!("Code placeholder not localized for lang \"{}\"", lang)
                }))
            } // TODO: is this placeholder satisfactory?
            // TODO: we could try extracting content from HTML nodes, I guess?
            Event::Html(_) => (),              // Ignore HTML, too
            Event::FootnoteReference(_) => (), // Ignore footnote references as well
            Event::SoftBreak => {
                assert_eq!(span.end - span.start, 1);
                mappings.push(StringMapping::new(&span, text.len()));
                text.push(' ');
            }
            Event::HardBreak => {
                assert_eq!(span.end - span.start, 1);
                mappings.push(StringMapping::new(&span, text.len()));
                text.push('\n');
            }
            Event::Rule => (),              // Ignore horizontal rules
            Event::TaskListMarker(_) => (), // Ignore task list markers

            Event::End(tag) => {
                if depth == 0 {
                    return Ok(tag);
                }
                depth -= 1;
            }
        }
    }
}

/// A scribe, responsible for outputting correction suggestions and keeping track of their count.
struct SuggestionScribe<'a> {
    output: &'a mut StandardStream,
    config: &'a Config,
    rules_config: &'a RulesConfig<'a>,

    nb_reports: usize,
}

impl<'a> SuggestionScribe<'a> {
    /// Creates a new scribe, writing to the given stream and with the given configuration.
    fn new(
        output: &'a mut StandardStream,
        config: &'a Config,
        rules_config: &'a RulesConfig<'a>,
    ) -> Self {
        Self {
            output,
            config,
            rules_config,

            nb_reports: 0,
        }
    }

    /// Reports a suggestion at a certain span within the given source file.
    /// The report's type (as well as whether it is output at all) is determined by the mappings
    /// given when creating the scribe.
    fn report<Name: Display + Clone, Source: AsRef<str>>(
        &mut self,
        suggestion: &Suggestion,
        files: &SimpleFile<Name, Source>,
        span: Range<usize>,
    ) {
        let severity: Option<_> = self.rules_config.get(suggestion.source()).into();
        let diagnostic = Diagnostic::new(severity.unwrap_or_else(|| {
            panic!(
                "Rule {} is disabled, but still produced a diagnostic",
                suggestion.source()
            )
        }))
        .with_code(suggestion.source())
        .with_message(suggestion.message())
        // `()` being [`SimpleFile`]'s [`FileId`][codespan_reporting::files::Files]
        .with_labels(vec![Label::primary((), span)])
        .with_notes(
            suggestion
                .replacements()
                .iter()
                .map(|repl| format!("help: consider replacing it with `{}`", repl))
                .collect(),
        );

        term::emit(&mut self.output.lock(), self.config, files, &diagnostic).unwrap();
        self.nb_reports += 1;
    }

    /// How many reports the scribe has output so far.
    fn nb_reports(&self) -> usize {
        self.nb_reports
    }
}

/// What kind of severity a rule can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticLevel {
    Off, // The rule is disabled entirely; this might speed up processing.
    Note,
    Warning,
    Error, // Any of these causes the renderer to exit unsuccessfully.
}

impl From<DiagnosticLevel> for Option<Severity> {
    fn from(level: DiagnosticLevel) -> Self {
        match level {
            DiagnosticLevel::Off => None,
            DiagnosticLevel::Note => Some(Severity::Note),
            DiagnosticLevel::Warning => Some(Severity::Warning),
            DiagnosticLevel::Error => Some(Severity::Error),
        }
    }
}

impl FromStr for DiagnosticLevel {
    type Err = ParseDiagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use DiagnosticLevel::*;

        Ok(if s == "off" {
            Off
        } else if s == "note" {
            Note
        } else if s == "warning" {
            Warning
        } else if s == "error" {
            Error
        } else {
            return Err(ParseDiagError);
        })
    }
}

/// An error obtained by attempting to parse a diagnostic from a string.
#[derive(Debug)]
struct ParseDiagError;

impl Display for ParseDiagError {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(
            fmt,
            "Expected one of \"off\", \"note\", \"warning\", or \"error\""
        )
    }
}

impl std::error::Error for ParseDiagError {}

/// Configuration on how to report rules' suggestions.
struct RulesConfig<'a>(HashMap<&'a str, DiagnosticLevel>);

impl<'a> RulesConfig<'a> {
    // WARNING: This will misbehave if there is "overlap", i.e. a generic rule and a more specific
    // rule covered by it!!
    // TODO: these are english rules... add different configuration per language!
    const DEFAULT_CONFIG: &'static [(&'static str, DiagnosticLevel)] = &[
        ("TYPOGRAPHY/EN_QUOTES", DiagnosticLevel::Off), // mdBook automatically generates smart quotes, so hush.
        // Mark typography requirements involving Unicode chars as simple notes
        ("TYPOGRAPHY/ARROWS", DiagnosticLevel::Note),
        ("TYPOGRAPHY/MULTIPLICATION_SIGN/0", DiagnosticLevel::Note),
        ("TYPOGRAPHY/TRADEMARK/0", DiagnosticLevel::Note),
        ("TYPOGRAPHY/R_SYMBOL/0", DiagnosticLevel::Note),
        ("PUNCTUATION/DASH_RULE", DiagnosticLevel::Note),
        ("TYPOGRAPHY/PLUS_MINUS/0", DiagnosticLevel::Note),
        ("TYPOGRAPHY/PLUS_MINUS/1", DiagnosticLevel::Note),
        // LanguageTool has those disabled by default
        // TODO: Auto-generate those
        // Note: more general rules are preferred when applicable, for easier cherry-picking,
        // except when there is only one or two sub-rules (because there would be little to no benefit to globbing)
        ("TYPOS/CAN_NOT/0", DiagnosticLevel::Off),
        ("TYPOS/FLASHPOINT/0", DiagnosticLevel::Off),
        ("GRAMMAR/NOT_SURE_IT_WORKS/0", DiagnosticLevel::Off),
        ("GRAMMAR/THAT_MISSING_VERB/0", DiagnosticLevel::Off),
        ("GRAMMAR/THAT_MISSING_VERB/1", DiagnosticLevel::Off),
        ("GRAMMAR/AGREEMENT_THEIR_HIS/0", DiagnosticLevel::Off),
        ("GRAMMAR/AGREEMENT_THEIR_HIS/1", DiagnosticLevel::Off),
        (
            "COLLOCATIONS/COLLOCATIONS_ERRORS_BOKOMARU/9",
            DiagnosticLevel::Off,
        ),
        (
            "COLLOCATIONS/COLLOCATIONS_ERRORS_BOKOMARU/21",
            DiagnosticLevel::Off,
        ),
        (
            "COLLOCATIONS/COLLOCATIONS_ERRORS_BOKOMARU/111",
            DiagnosticLevel::Off,
        ),
        (
            "COLLOCATIONS/COLLOCATIONS_ERRORS_BOKOMARU/143",
            DiagnosticLevel::Off,
        ),
        (
            "PUNCTUATION/COMMA_COMPOUND_SENTENCE_2/2",
            DiagnosticLevel::Off,
        ),
        ("PUNCTUATION/COMMA_AND_NUMBERS/1", DiagnosticLevel::Off),
        ("PUNCTUATION/COMMA_BEFORE_AND/0", DiagnosticLevel::Off),
        ("PUNCTUATION/SERIAL_COMMA_ON", DiagnosticLevel::Off),
        ("PUNCTUATION/SERIAL_COMMA_OFF/0", DiagnosticLevel::Off),
        ("PUNCTUATION/COMMA_WHICH/0", DiagnosticLevel::Off),
        (
            "CONFUSED_WORDS/COMPLAINS_COMPLAINTS/2",
            DiagnosticLevel::Off,
        ),
        ("NONSTANDARD_PHRASES/I_VE_A/0", DiagnosticLevel::Off),
        ("NONSTANDARD_PHRASES/I_VE_A/1", DiagnosticLevel::Off),
        ("REDUNDANCY/WORLD_AROUND_IT/0", DiagnosticLevel::Off),
        ("REDUNDANCY/VERY_SMALL_TINY/0", DiagnosticLevel::Off),
        ("REDUNDANCY/AND_ALSO/0", DiagnosticLevel::Off),
        ("REDUNDANCY/ARE_ABLE_TO/0", DiagnosticLevel::Off),
        ("REDUNDANCY/BEST_EVER/0", DiagnosticLevel::Off),
        ("REDUNDANCY/BLEND_TOGETHER/0", DiagnosticLevel::Off),
        ("REDUNDANCY/BRIEF_MOMENT/0", DiagnosticLevel::Off),
        ("REDUNDANCY/NEGATE_MEANING", DiagnosticLevel::Off),
        ("REDUNDANCY/EXACT_SAME/0", DiagnosticLevel::Off),
        ("REDUNDANCY/EXACT_SAME/1", DiagnosticLevel::Off),
        ("REDUNDANCY/IN_ORDER_TO/0", DiagnosticLevel::Off),
        ("STYLE/ALSO_SENT_END/0", DiagnosticLevel::Off),
        ("PLAIN_ENGLISH", DiagnosticLevel::Off),
        ("WIKIPEDIA", DiagnosticLevel::Off),
    ];

    pub fn new(user_config: Option<&'a Table>) -> Result<Self> {
        let default: HashMap<_, _> = Self::DEFAULT_CONFIG.iter().cloned().collect();

        // If no user config was specified at all, use the default config as-is
        let config = match user_config {
            None => default,
            Some(user_config) => {
                // Overlay the user's config on top of the default config.
                // The tricky bit is that we want the user's "broader" settings to override the
                // "default" ones.
                // This is achieved by actually generating an all-new config from the user's, then
                // applying the non-overridden default rules.
                let mut config = user_config
                    .iter()
                    .map(|(name, val)| {
                        val.as_str()
                            .ok_or_else(|| {
                                Error::msg(format!(
                                    "Expected a string for \"{}\", not {}",
                                    name,
                                    val.type_str()
                                ))
                            })
                            // We cannot return the Result directly, as the error types are different
                            // `?` performs a conversion, so it's fine
                            .and_then(|level| {
                                Ok((
                                    name.as_str(),
                                    level
                                        .parse()
                                        .context(format!("Got \"{}\" for \"{}\"", level, name))?,
                                ))
                            })
                    })
                    .collect::<Result<HashMap<_, _>, _>>()?;

                for (name, level) in default.iter() {
                    let mut end = name.len();
                    loop {
                        // Is the rule covered, either directly or by a wider rule?
                        // (Note that this assumes that no two default rules overlap, since HashMap
                        // can and will scramble insertion order)
                        if config.get(&name[..end]).is_some() {
                            break;
                        }
                        // Otherwise, keep trying at the next slash position (from the right), and
                        // exit while inserting the rule if no mo'
                        if let Some(pos) = name[..end].rfind('/') {
                            end = pos;
                        } else {
                            config.insert(name, *level);
                            break;
                        }
                    }
                }

                config
            }
        };

        Ok(Self(config))
    }

    fn try_get(&self, name: &str) -> Option<DiagnosticLevel> {
        // Rules are normally formatted as CATEGORY/GROUP/RULE, so obtain the position of the two slashes
        let mut slash_pos = [0, 0];
        let res = name.match_indices('/').try_fold(0, |n, (pos, _)| {
            *slash_pos.get_mut(n)? = pos;
            Some(n + 1)
        });
        // If we didn't get exactly two slashes, the provided name is invalid
        if res != Some(2) {
            panic!("Bad rule name \"{}\"", name);
        }

        self.0
            .get(name)
            // Not found, check if a "level 2" generic rule applies instead
            .or_else(|| self.0.get(&name[..slash_pos[1]]))
            // If still not found, try with a "top-level" rule
            .or_else(|| self.0.get(&name[..slash_pos[0]]))
            .copied()
    }

    pub fn get(&self, name: &str) -> DiagnosticLevel {
        self.try_get(name)
            // Still not? Then default the rule to warning level
            .unwrap_or(DiagnosticLevel::Warning)
    }
}

impl SpellChecker {
    fn load_checker(lang: &str, config: &RulesConfig<'_>) -> Result<(Tokenizer, Rules)> {
        // Read nlprule's binaries to configure tokenizer and rules
        let tokenizer = Tokenizer::new(Path::new(env!("OUT_DIR")).join(tokenizer_filename(lang)))
            .context("Failed to read tokenizer binary")?;
        let mut rules = Rules::new(Path::new(env!("OUT_DIR")).join(rules_filename(lang)))
            .context("Failed to read rules binary")?;

        // Speed up processing by disabling the rules that were... er, disabled!
        for rule in rules.rules_mut() {
            if config.get(&rule.id().to_string()) == DiagnosticLevel::Off {
                rule.disable();
            } else {
                rule.enable(); // Rules may not be enabled by nlprules by default
            }
        }

        Ok((tokenizer, rules))
    }
}

/// Returns to enable color if and only if a stream is a TTY.
fn auto_color(stream: atty::Stream) -> ColorChoice {
    if atty::is(stream) {
        ColorChoice::Auto
    } else {
        ColorChoice::Never
    }
}

/// Shifts a range right by some amount.
fn shift<T: Add<U> + Copy, U: Copy>(range: &Range<T>, shift: U) -> Range<<T as Add<U>>::Output> {
    (range.start + shift)..(range.end + shift)
}

/// Is this Markdown tag considered inline?
fn is_inline(tag: &Tag) -> bool {
    use Tag::*;

    match tag {
        Paragraph
        | Heading(..)
        | BlockQuote
        | CodeBlock(..)
        | List(..)
        | Item
        | FootnoteDefinition(..)
        | Table(..)
        | TableHead
        | TableRow
        | TableCell => false,
        Emphasis | Strong | Strikethrough | Link(..) | Image(..) => true,
    }
}

/// Given a span into a "mapped" string, find the corresponding span in the original string.
///
/// The "mapped" string is basically a collection of slices of the original string.
/// Given a span within that "mapped" string, find the span in the original string that encompasses
/// exactly the text in the mapped string's slice.
fn find_orig_span(span: &Range<usize>, offsets: &[StringMapping]) -> Result<Range<usize>> {
    let mut spans = offsets
        .iter()
        .skip_while(|mapping| {
            // Skip while the span's end is before our start
            mapping.dest_start + mapping.len < span.start
        })
        .take_while(|mapping| {
            // Keep spans until they begin after our end
            mapping.dest_start < span.end
        });

    // Grab the corresponding starting & ending positions within the original string
    let start = spans
        .next()
        .expect("Suggestion given outside of any mappings??");
    if start.dest_start > span.start {
        return Err(Error::msg(format!(
            "Earliest span starts at {}, we want {}",
            start.dest_start, span.start
        )));
    }
    let start_pos = start.orig_start + (span.start - start.dest_start);

    let end = spans.last().unwrap_or(start);
    if end.dest_start + end.len < span.end {
        return Err(Error::msg(format!(
            "Last span ends at {}, we want {}",
            end.dest_start + end.len,
            span.end
        )));
    }
    let end_pos = end.orig_start + (span.end - end.dest_start);

    Ok(start_pos..end_pos)
}

#[derive(Debug, PartialEq)]
/// A mapping of some bytes from string A to string B; that is,
/// `&orig[orig_start..orig_start + len] == &dest[dest_start..dest_start + len()]`.
struct StringMapping {
    /// Offset within the original string.
    orig_start: usize,
    /// Offset within the destination string.
    dest_start: usize,
    /// Length of the mapping.
    len: usize,
}

impl StringMapping {
    fn new(orig_range: &Range<usize>, dest_start: usize) -> Self {
        Self {
            orig_start: orig_range.start,
            dest_start,
            len: orig_range.end - orig_range.start,
        }
    }
}

/// Keeps track of whether an error occurred in a given part.
struct Part<'a> {
    /// The part's type.
    part_type: PartType<'a>,
    /// Has an error occurred within that part so far?
    has_err: bool,
}

impl<'a> Part<'a> {
    fn new(part_type: PartType<'a>) -> Self {
        Self {
            part_type,
            has_err: false,
        }
    }

    /// Reports the part's name, unless an error already occurred.
    fn report(&mut self) {
        if !self.has_err {
            self.has_err = true;
            let mut stderr = StandardStream::stderr(auto_color(atty::Stream::Stderr));

            stderr.set_color(ColorSpec::new().set_bold(true)).unwrap();
            match self.part_type {
                PartType::Prefix => writeln!(&mut stderr, "In prefix part:"),
                PartType::Part(s) => writeln!(&mut stderr, "In part \"{}\":", s),
                PartType::Suffix => writeln!(&mut stderr, "In suffix part:"),
            }
            .unwrap();
            stderr.reset().unwrap();
        }
    }
}

/// A part type: "regular" (with a name), or prefix / suffix.
#[derive(Debug)]
enum PartType<'a> {
    Prefix,
    Part(&'a str),
    Suffix,
}

/// A summary item, and its location within the summary file.
#[derive(Debug)]
enum SummaryItem {
    Part(String, Vec<StringMapping>),
    Chapter(String, Vec<StringMapping>),
    SuffixPart,
}

/// Parses the summary file to extract the location at which its different items are defined.
///
/// This includes the otherwise unreported "suffix" (pseudo-)part.
fn parse_summary<'a>(
    summary: &'a str,
    mut get_block_text: impl FnMut(&mut OffsetIter<'a>) -> Result<(String, Vec<StringMapping>, Tag<'a>)>,
) -> Result<Vec<SummaryItem>> {
    use SummaryLocsState::*;

    // This parsing process attempts to mimic the way `mdbook` itself parses `SUMMARY.md`.
    // *Obviously*, there will likely be discrepancies, as `mdbook` is more lax than its spec,
    // but any sanely-written summaries should work regardless.
    // Note that no attempt is made to handle data `mdbook` would reject, as then this renderer
    // wouldn't be called anyway; thus, expect a lot of `unwrap`/`expect`s.
    //
    // Please note that this behavior was not RE'd from the mdbook code, but deduced by feeding
    // it various input files.
    let mut events = Parser::new(summary).into_offset_iter();
    let mut state = Prefix;
    let mut items = Vec::new();

    while let Some((evt, _)) = events.next() {
        match state {
            Prefix => {
                // For prefix chapters, mdBook parses every link as a chapter, and ignores the rest
                if let Event::Start(Tag::Link(ty, url, ti)) = evt {
                    let (name, mappings, ending_tag) = get_block_text(&mut events).context(
                        format!("Error within {:?} link to {} (title {})", ty, url, ti),
                    )?;
                    assert_eq!(ending_tag, Tag::Link(ty, url, ti));

                    items.push(SummaryItem::Chapter(name, mappings));
                }
                // mdBook ignores non-level 1 headings
                else if let Event::Start(Tag::Heading(1)) = evt {
                    let (name, mappings, ending_tag) =
                        get_block_text(&mut events).context("Error within top-level heading")?;
                    assert_eq!(ending_tag, Tag::Heading(1));

                    items.push(SummaryItem::Part(name, mappings));

                    state = AfterHeader;
                }
            }

            // Then, after each header, mdBook ignores everything up until the first list
            // TODO: is the list's type ignored?
            AfterHeader => {
                // Every list item is processed, regardless of whether part of a single list or several (separated via HTML comments)
                if let Event::Start(Tag::List(_)) = evt {
                    state = Normal(0);
                }
                // If the last list is followed by a heading, process the next part
                else if let Event::Start(Tag::Heading(1)) = evt {
                    let (name, mappings, ending_tag) =
                        get_block_text(&mut events).context("Error within top-level heading")?;
                    assert_eq!(ending_tag, Tag::Heading(1));

                    items.push(SummaryItem::Part(name, mappings));
                }
                // Otherwise, process suffix chapters
                else {
                    items.push(SummaryItem::SuffixPart);

                    state = Suffix;
                }
            }

            Normal(depth) => {
                if let Event::Start(Tag::Item) = evt {
                    // Each item must start with a link...
                    let tag = match events.next() {
                        Some((Event::Start(tag @ Tag::Link(..)), _)) => tag,
                        evt => panic!("Unexpected {:?}", evt),
                    };

                    let (name, mappings, ending_tag) = get_block_text(&mut events)
                        .context(format!("Error within link {:?}", tag))?;
                    assert_eq!(ending_tag, tag);

                    items.push(SummaryItem::Chapter(name, mappings));

                    // ...anything after that is ignored, up to the end of the item, or the
                    // beginning of a sub-list
                    // We can't use `for` because it consumes the iterator
                    #[allow(clippy::while_let_on_iterator)]
                    while let Some((evt, _)) = events.next() {
                        match evt {
                            Event::End(Tag::Item) => break,
                            Event::Start(Tag::Item) => unreachable!(),
                            Event::Start(Tag::List(_)) => {
                                state = Normal(depth + 1);
                                break;
                            }
                            _ => (),
                        }
                    }
                }
                // Track list depth
                else if let Event::End(Tag::List(_)) = evt {
                    state = if depth == 0 {
                        AfterHeader
                    } else {
                        Normal(depth - 1)
                    };
                } else if let Event::Start(Tag::List(_)) = evt {
                    state = Normal(depth + 1);
                }
            }

            Suffix => {
                if let Event::Start(Tag::Link(ty, url, ti)) = evt {
                    let (name, mappings, ending_tag) = get_block_text(&mut events).context(
                        format!("Error within {:?} link to {} (title {})", ty, url, ti),
                    )?;
                    assert_eq!(ending_tag, Tag::Link(ty, url, ti));

                    items.push(SummaryItem::Chapter(name, mappings));
                }
            }
        }
    }

    Ok(items)
}

/// Which part of the summary is currently being parsed: prefix chapters, "normal" parts, or suffix
/// chapters?
#[derive(Debug)]
enum SummaryLocsState {
    Prefix,
    AfterHeader,
    Normal(usize),
    Suffix,
}
