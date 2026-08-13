/**
 * The element helper, shared by `render.js` and `md.js`.
 *
 * It lives in its own module so those two can both use it without importing each other —
 * ES modules tolerate a cycle between them, but only by accident of hoisting, and this is
 * the seam where model output becomes DOM. Worth keeping boring.
 *
 * Everything sets `textContent`; nothing here assigns `innerHTML`, so escaping is a
 * property of construction rather than something each call site must remember.
 */

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }

  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }

  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}
