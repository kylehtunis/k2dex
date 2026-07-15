// react-select menus portal to <body> so they escape overflow containers.
// During build-time prerendering (renderToString in Node) there is no
// document; undefined keeps the menu inline there, which is invisible in the
// static HTML since no menu is ever open at render time.
export const MENU_PORTAL_TARGET: HTMLElement | undefined =
  typeof document === "undefined" ? undefined : document.body;
