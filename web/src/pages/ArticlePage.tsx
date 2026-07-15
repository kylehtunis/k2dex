// /articles/:slug — resolves the slug against the article registry and renders
// that article's body inside a light shell (a "back to Articles" link). An
// unknown slug redirects to the index. The article component itself owns its
// full layout (e.g. SciencePage renders its own header).

import { Suspense } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { ARTICLE_COMPONENTS } from "../articles/components";
import { ARTICLES } from "../articles/articles";

export function ArticlePage() {
  const { slug = "" } = useParams();
  const meta = ARTICLES.find((a) => a.slug === slug);
  const Body = ARTICLE_COMPONENTS[slug];
  if (!meta || !Body) return <Navigate to="/articles/" replace />;
  return (
    <div className="lab-article">
      <nav className="lab-article-nav">
        <Link to="/articles/" className="lab-article-back">
          ← Articles
        </Link>
      </nav>
      <Suspense
        fallback={<p style={{ color: "var(--lab-ink-muted)" }}>Loading article…</p>}
      >
        <Body />
      </Suspense>
    </div>
  );
}
export default ArticlePage;
