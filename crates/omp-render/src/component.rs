use crate::out::{Out, OutError};
use crate::richtext::{Color, RichText, SemanticColor, Style};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ComponentKind {
    Box,
    Row,
    Col,
    Text,
    Pre,
    Hr,
    Badge,
    Callout,
    Link,
    Icon,
    Image,
    Usage,
    SemanticMessage,
    SemanticTool,
    SemanticSubagent,
    SemanticApproval,
    SemanticDirector,
    SemanticArtifact,
    SemanticAutoQA,
}

impl ComponentKind {
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Hr | Self::Icon | Self::Image)
    }

    pub const fn is_block(&self) -> bool {
        matches!(
            self,
            Self::Box
                | Self::Row
                | Self::Col
                | Self::Pre
                | Self::Callout
                | Self::SemanticMessage
                | Self::SemanticTool
                | Self::SemanticSubagent
                | Self::SemanticApproval
                | Self::SemanticDirector
                | Self::SemanticArtifact
                | Self::SemanticAutoQA
        )
    }

    pub const fn is_inline(&self) -> bool {
        matches!(
            self,
            Self::Text | Self::Badge | Self::Link | Self::Icon | Self::Usage
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComponentProps {
    pub values: BTreeMap<String, serde_json::Value>,
}

impl ComponentProps {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    pub fn set(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(val) = serde_json::to_value(value) {
            self.values.insert(key.into(), val);
        }
        self
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.values.get(key).and_then(|v| v.as_str())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.values.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.values.get(key).and_then(|v| v.as_bool())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ComponentValidationError {
    #[error("leaf element {0:?} cannot have children")]
    LeafCannotHaveChildren(ComponentKind),
    #[error("inline element {parent:?} cannot contain block child {child:?}: {reason}")]
    InvalidChild {
        parent: ComponentKind,
        child: ComponentKind,
        reason: &'static str,
    },
    #[error("element {element:?} missing required property '{prop}'")]
    MissingRequiredProp {
        element: ComponentKind,
        prop: &'static str,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub kind: ComponentKind,
    pub props: ComponentProps,
    pub children: Vec<Component>,
    pub text: Option<RichText>,
}

impl Component {
    pub fn new(
        kind: ComponentKind,
        props: ComponentProps,
        children: Vec<Component>,
    ) -> Result<Self, ComponentValidationError> {
        let comp = Self {
            kind,
            props,
            children,
            text: None,
        };
        comp.validate()?;
        Ok(comp)
    }

    pub fn text(rich: RichText) -> Self {
        Self {
            kind: ComponentKind::Text,
            props: ComponentProps::new(),
            children: Vec::new(),
            text: Some(rich),
        }
    }

    pub fn plain_text(text: impl Into<String>) -> Self {
        Self::text(RichText::from_plain(text))
    }

    pub fn pre(content: impl Into<String>) -> Self {
        let mut props = ComponentProps::new();
        props = props.set("code", true);
        Self {
            kind: ComponentKind::Pre,
            props,
            children: Vec::new(),
            text: Some(RichText::from_plain(content)),
        }
    }

    pub fn hr() -> Self {
        Self {
            kind: ComponentKind::Hr,
            props: ComponentProps::new(),
            children: Vec::new(),
            text: None,
        }
    }

    pub fn badge(variant: SemanticColor, label: impl Into<String>) -> Self {
        let label_str = label.into();
        let mut props = ComponentProps::new();
        props = props.set("label", &label_str);
        props = props.set("variant", format!("{variant:?}"));
        let mut rt = RichText::new();
        rt.push(
            Style::new().fg(Color::Semantic(variant)).bold(),
            format!("[{label_str}]"),
        );
        Self {
            kind: ComponentKind::Badge,
            props,
            children: Vec::new(),
            text: Some(rt),
        }
    }

    pub fn callout(
        variant: SemanticColor,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<Self, ComponentValidationError> {
        let mut props = ComponentProps::new();
        props = props.set("variant", format!("{variant:?}"));
        props = props.set("title", title.into());
        let mut rt = RichText::new();
        rt.push(Style::semantic(variant), body.into());
        let body_comp = Self::text(rt);
        Self::new(ComponentKind::Callout, props, vec![body_comp])
    }

    pub fn link(href: impl Into<String>, label: impl Into<String>) -> Self {
        let href_str = href.into();
        let label_str = label.into();
        let mut props = ComponentProps::new();
        props = props.set("href", &href_str);
        let mut rt = RichText::new();
        rt.push(
            Style::new()
                .fg(Color::Semantic(SemanticColor::Primary))
                .underline(),
            label_str,
        );
        Self {
            kind: ComponentKind::Link,
            props,
            children: Vec::new(),
            text: Some(rt),
        }
    }

    pub fn icon(name: impl Into<String>) -> Self {
        let name_str = name.into();
        let mut props = ComponentProps::new();
        props = props.set("name", &name_str);
        let mut rt = RichText::new();
        rt.push_plain(format!(":{name_str}:"));
        Self {
            kind: ComponentKind::Icon,
            props,
            children: Vec::new(),
            text: Some(rt),
        }
    }

    pub fn image(src: impl Into<String>, alt: impl Into<String>, width: u32, height: u32) -> Self {
        let mut props = ComponentProps::new();
        props = props.set("src", src.into());
        props = props.set("alt", alt.into());
        props = props.set("width", width);
        props = props.set("height", height);
        Self {
            kind: ComponentKind::Image,
            props,
            children: Vec::new(),
            text: None,
        }
    }

    pub fn usage(prompt_tokens: u64, completion_tokens: u64, cost: Option<f64>) -> Self {
        let mut props = ComponentProps::new();
        props = props.set("prompt_tokens", prompt_tokens);
        props = props.set("completion_tokens", completion_tokens);
        if let Some(c) = cost {
            props = props.set("cost", c);
        }
        let total = prompt_tokens + completion_tokens;
        let mut rt = RichText::new();
        rt.push(
            Style::semantic(SemanticColor::Muted),
            format!("tokens: {total} (prompt: {prompt_tokens}, comp: {completion_tokens})"),
        );
        Self {
            kind: ComponentKind::Usage,
            props,
            children: Vec::new(),
            text: Some(rt),
        }
    }

    pub fn box_container(
        props: ComponentProps,
        children: Vec<Component>,
    ) -> Result<Self, ComponentValidationError> {
        Self::new(ComponentKind::Box, props, children)
    }

    pub fn row(children: Vec<Component>) -> Result<Self, ComponentValidationError> {
        Self::new(ComponentKind::Row, ComponentProps::new(), children)
    }

    pub fn col(children: Vec<Component>) -> Result<Self, ComponentValidationError> {
        Self::new(ComponentKind::Col, ComponentProps::new(), children)
    }

    pub fn validate(&self) -> Result<(), ComponentValidationError> {
        if self.kind.is_leaf() && !self.children.is_empty() {
            return Err(ComponentValidationError::LeafCannotHaveChildren(self.kind));
        }

        if self.kind.is_inline() {
            for child in &self.children {
                if child.kind.is_block() {
                    return Err(ComponentValidationError::InvalidChild {
                        parent: self.kind,
                        child: child.kind,
                        reason: "inline component cannot contain block child",
                    });
                }
            }
        }

        match self.kind {
            ComponentKind::Pre => {
                for child in &self.children {
                    if child.kind.is_block() || child.kind == ComponentKind::Pre {
                        return Err(ComponentValidationError::InvalidChild {
                            parent: self.kind,
                            child: child.kind,
                            reason: "Pre component cannot contain block or nested Pre child",
                        });
                    }
                }
            }
            ComponentKind::Link => {
                if self.props.get_str("href").is_none() {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "href",
                    });
                }
                for child in &self.children {
                    if child.kind == ComponentKind::Link {
                        return Err(ComponentValidationError::InvalidChild {
                            parent: self.kind,
                            child: child.kind,
                            reason: "Link component cannot contain nested Link",
                        });
                    }
                }
            }
            ComponentKind::Callout => {
                if self.props.get_str("title").is_none() {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "title",
                    });
                }
                if self.props.get_str("variant").is_none() {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "variant",
                    });
                }
                for child in &self.children {
                    if child.kind == ComponentKind::Callout {
                        return Err(ComponentValidationError::InvalidChild {
                            parent: self.kind,
                            child: child.kind,
                            reason: "Callout component cannot contain nested Callout",
                        });
                    }
                }
            }
            ComponentKind::Badge => {
                if self.props.get_str("label").is_none() {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "label",
                    });
                }
            }
            ComponentKind::Icon => {
                if self.props.get_str("name").is_none() {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "name",
                    });
                }
            }
            ComponentKind::Image
                if self.props.get_str("src").is_none() => {
                    return Err(ComponentValidationError::MissingRequiredProp {
                        element: self.kind,
                        prop: "src",
                    });
                }
            _ => {}
        }

        for child in &self.children {
            child.validate()?;
        }

        Ok(())
    }

    fn render_inline<O: Out>(&self, out: &mut O) -> Result<(), OutError> {
        if let Some(text) = &self.text {
            out.write_rich(text)?;
        }
        for child in &self.children {
            child.render_inline(out)?;
        }
        Ok(())
    }

    pub fn render_to_sink<O: Out>(&self, out: &mut O, indent: usize) -> Result<(), OutError> {
        let pad = "  ".repeat(indent);

        match self.kind {
            ComponentKind::Box => {
                let border = self.props.get_bool("border").unwrap_or(false);
                let title = self.props.get_str("title");

                if border {
                    let mut top = format!("{pad}+---");
                    if let Some(t) = title {
                        top.push_str(&format!(" [ {t} ] ---"));
                    }
                    top.push('+');
                    out.write_plain(&top)?;
                    out.line_break()?;
                }

                for child in &self.children {
                    child.render_to_sink(out, indent + if border { 1 } else { 0 })?;
                }

                if border {
                    out.write_plain(&format!("{pad}+------------------------+"))?;
                    out.line_break()?;
                }
            }
            ComponentKind::Row => {
                out.write_plain(&pad)?;
                for (i, child) in self.children.iter().enumerate() {
                    if i > 0 {
                        out.write_plain(" ")?;
                    }
                    if child.kind.is_inline() {
                        child.render_inline(out)?;
                    } else {
                        child.render_to_sink(out, indent)?;
                    }
                }
                out.line_break()?;
            }
            ComponentKind::Col => {
                for child in &self.children {
                    child.render_to_sink(out, indent)?;
                }
            }
            ComponentKind::Text => {
                if let Some(rt) = &self.text {
                    if !pad.is_empty() {
                        out.write_plain(&pad)?;
                    }
                    for run in &rt.runs {
                        let mut parts = run.text.split('\n').peekable();
                        while let Some(part) = parts.next() {
                            out.write_str(run.style, part)?;
                            if parts.peek().is_some() {
                                out.line_break()?;
                                if !pad.is_empty() {
                                    out.write_plain(&pad)?;
                                }
                            }
                        }
                    }
                    out.line_break()?;
                } else {
                    if !pad.is_empty() {
                        out.write_plain(&pad)?;
                    }
                    self.render_inline(out)?;
                    out.line_break()?;
                }
            }
            ComponentKind::Pre => {
                out.write_plain(&format!("{pad}```"))?;
                out.line_break()?;
                if let Some(rt) = &self.text {
                    out.write_plain(&pad)?;
                    for run in &rt.runs {
                        let mut parts = run.text.split('\n').peekable();
                        while let Some(part) = parts.next() {
                            out.write_str(run.style, part)?;
                            if parts.peek().is_some() {
                                out.line_break()?;
                                out.write_plain(&pad)?;
                            }
                        }
                    }
                    out.line_break()?;
                }
                out.write_plain(&format!("{pad}```"))?;
                out.line_break()?;
            }
            ComponentKind::Hr => {
                out.write_plain(&format!("{pad}----------------------------------------"))?;
                out.line_break()?;
            }
            ComponentKind::Badge => {
                if let Some(rt) = &self.text {
                    out.write_rich(rt)?;
                }
            }
            ComponentKind::Callout => {
                let title = self.props.get_str("title").unwrap_or("Notice");
                let color = match self.props.get_str("variant").unwrap_or("Info") {
                    "Error" => SemanticColor::Error,
                    "Warning" => SemanticColor::Warning,
                    "Muted" => SemanticColor::Muted,
                    "Success" => SemanticColor::Success,
                    _ => SemanticColor::Info,
                };
                out.write_str(
                    Style::new().bold().fg(Color::Semantic(color)),
                    &format!("{pad}> {title}:"),
                )?;
                out.line_break()?;
                for child in &self.children {
                    child.render_to_sink(out, indent + 1)?;
                }
            }
            ComponentKind::Link => {
                if let Some(rt) = &self.text {
                    out.write_rich(rt)?;
                }
            }
            ComponentKind::Icon => {
                if let Some(rt) = &self.text {
                    out.write_rich(rt)?;
                }
            }
            ComponentKind::Image => {
                let alt = self.props.get_str("alt").unwrap_or("image");
                let src = self.props.get_str("src").unwrap_or("");
                out.write_plain(&format!("{pad}[img: {alt} ({src})]"))?;
                out.line_break()?;
            }
            ComponentKind::Usage => {
                if let Some(rt) = &self.text {
                    if !pad.is_empty() {
                        out.write_plain(&pad)?;
                    }
                    out.write_rich(rt)?;
                    out.line_break()?;
                }
            }
            _ => {
                for child in &self.children {
                    child.render_to_sink(out, indent)?;
                }
            }
        }

        Ok(())
    }
}
