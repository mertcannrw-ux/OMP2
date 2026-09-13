use omp_render::{
    Component, ComponentKind, ComponentProps, ComponentValidationError, SemanticColor,
};

#[test]
fn test_leaf_cannot_have_children() {
    let child = Component::plain_text("child");

    // Hr is a leaf
    let hr_err = Component::new(
        ComponentKind::Hr,
        ComponentProps::new(),
        vec![child.clone()],
    )
    .unwrap_err();
    assert_eq!(
        hr_err,
        ComponentValidationError::LeafCannotHaveChildren(ComponentKind::Hr)
    );

    // Icon is a leaf
    let mut icon_props = ComponentProps::new();
    icon_props = icon_props.set("name", "check");
    let icon_err =
        Component::new(ComponentKind::Icon, icon_props, vec![child.clone()]).unwrap_err();
    assert_eq!(
        icon_err,
        ComponentValidationError::LeafCannotHaveChildren(ComponentKind::Icon)
    );

    // Image is a leaf
    let mut img_props = ComponentProps::new();
    img_props = img_props.set("src", "https://example.com/logo.png");
    let img_err = Component::new(ComponentKind::Image, img_props, vec![child]).unwrap_err();
    assert_eq!(
        img_err,
        ComponentValidationError::LeafCannotHaveChildren(ComponentKind::Image)
    );
}

#[test]
fn test_inline_component_cannot_contain_block_child() {
    let block_child = Component::box_container(ComponentProps::new(), vec![]).unwrap();

    // Text cannot contain Box
    let text_err = Component::new(
        ComponentKind::Text,
        ComponentProps::new(),
        vec![block_child.clone()],
    )
    .unwrap_err();
    assert!(matches!(
        text_err,
        ComponentValidationError::InvalidChild { .. }
    ));

    // Badge cannot contain Box
    let mut badge_props = ComponentProps::new();
    badge_props = badge_props.set("label", "Admin");
    let badge_err =
        Component::new(ComponentKind::Badge, badge_props, vec![block_child.clone()]).unwrap_err();
    assert!(matches!(
        badge_err,
        ComponentValidationError::InvalidChild { .. }
    ));

    // Link cannot contain Box
    let mut link_props = ComponentProps::new();
    link_props = link_props.set("href", "https://example.com");
    let link_err = Component::new(ComponentKind::Link, link_props, vec![block_child]).unwrap_err();
    assert!(matches!(
        link_err,
        ComponentValidationError::InvalidChild { .. }
    ));
}

#[test]
fn test_nested_links_and_pre_rejected() {
    // Nested Link in Link is forbidden
    let inner_link = Component::link("https://inner.com", "Inner");
    let mut outer_props = ComponentProps::new();
    outer_props = outer_props.set("href", "https://outer.com");
    let link_err = Component::new(ComponentKind::Link, outer_props, vec![inner_link]).unwrap_err();
    assert!(matches!(
        link_err,
        ComponentValidationError::InvalidChild {
            parent: ComponentKind::Link,
            child: ComponentKind::Link,
            ..
        }
    ));

    // Pre containing Pre is forbidden
    let inner_pre = Component::pre("code block");
    let pre_err =
        Component::new(ComponentKind::Pre, ComponentProps::new(), vec![inner_pre]).unwrap_err();
    assert!(matches!(
        pre_err,
        ComponentValidationError::InvalidChild {
            parent: ComponentKind::Pre,
            child: ComponentKind::Pre,
            ..
        }
    ));

    // Callout containing Callout is forbidden
    let inner_callout = Component::callout(SemanticColor::Info, "Inner", "Text").unwrap();
    let mut callout_props = ComponentProps::new();
    callout_props = callout_props.set("title", "Outer");
    callout_props = callout_props.set("variant", "Info");
    let callout_err =
        Component::new(ComponentKind::Callout, callout_props, vec![inner_callout]).unwrap_err();
    assert!(matches!(
        callout_err,
        ComponentValidationError::InvalidChild {
            parent: ComponentKind::Callout,
            child: ComponentKind::Callout,
            ..
        }
    ));
}

#[test]
fn test_missing_required_properties() {
    // Image missing src
    let img_err = Component::new(ComponentKind::Image, ComponentProps::new(), vec![]).unwrap_err();
    assert_eq!(
        img_err,
        ComponentValidationError::MissingRequiredProp {
            element: ComponentKind::Image,
            prop: "src",
        }
    );

    // Link missing href
    let link_err = Component::new(ComponentKind::Link, ComponentProps::new(), vec![]).unwrap_err();
    assert_eq!(
        link_err,
        ComponentValidationError::MissingRequiredProp {
            element: ComponentKind::Link,
            prop: "href",
        }
    );

    // Icon missing name
    let icon_err = Component::new(ComponentKind::Icon, ComponentProps::new(), vec![]).unwrap_err();
    assert_eq!(
        icon_err,
        ComponentValidationError::MissingRequiredProp {
            element: ComponentKind::Icon,
            prop: "name",
        }
    );

    // Badge missing label
    let badge_err =
        Component::new(ComponentKind::Badge, ComponentProps::new(), vec![]).unwrap_err();
    assert_eq!(
        badge_err,
        ComponentValidationError::MissingRequiredProp {
            element: ComponentKind::Badge,
            prop: "label",
        }
    );
}

#[test]
fn test_valid_components_pass_validation() {
    let text = Component::plain_text("Hello");
    let link = Component::link("https://example.com", "Click");
    let badge = Component::badge(SemanticColor::Success, "OK");
    let hr = Component::hr();

    let row = Component::row(vec![text, link, badge, hr]).unwrap();
    assert_eq!(row.kind, ComponentKind::Row);
    assert_eq!(row.children.len(), 4);
}
