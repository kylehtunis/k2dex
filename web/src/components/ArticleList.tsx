// Shared article-card list, driven by the ARTICLES registry (newest first).
// Rendered both by the /articles index and by the home page's Articles section,
// so the two stay in sync as articles are added.

import { Link } from "react-router-dom";
import { articlePath, articlesByDate, formatArticleDate } from "../articles/articles";

export function ArticleList() {
  const articles = articlesByDate();
  return (
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
  );
}
export default ArticleList;
