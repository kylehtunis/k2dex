// /articles — index of write-ups (insights, learnings, implementation notes).
// Cards are driven by the ARTICLES registry, newest first. The first (and for
// now only) article is "The Science of k2dex".

import { Link } from "react-router-dom";
import { articlePath, articlesByDate, formatArticleDate } from "../articles/articles";

export function ArticlesPage() {
  const articles = articlesByDate();
  return (
    <div className="lab-articles">
      <header className="lab-articles-header">
        <h1>Articles</h1>
        <p className="lab-articles-lede">
          Write-ups on the ideas, findings, and implementation details behind k2dex.
        </p>
      </header>
      <ul className="lab-articles-list">
        {articles.map((a) => (
          <li key={a.slug}>
            <Link to={`/${articlePath(a.slug)}/`} className="lab-article-card">
              <div className="lab-article-card-body">
                <div className="lab-article-card-date">{formatArticleDate(a.date)}</div>
                <div className="lab-article-card-title">{a.title}</div>
                <p className="lab-article-card-desc">{a.description}</p>
              </div>
              <span className="lab-article-card-arrow">→</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
export default ArticlesPage;
