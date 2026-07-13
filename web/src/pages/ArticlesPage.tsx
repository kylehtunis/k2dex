// /articles — index of write-ups (insights, learnings, implementation notes).
// Cards are driven by the ARTICLES registry, newest first (see ArticleList,
// shared with the home page's Articles section).

import { ArticleList } from "../components/ArticleList";

export function ArticlesPage() {
  return (
    <div className="lab-articles">
      <header className="lab-articles-header">
        <h1>Articles</h1>
        <p className="lab-articles-lede">
          Write-ups on the ideas, findings, and implementation details behind k2dex.
        </p>
      </header>
      <ArticleList />
    </div>
  );
}
export default ArticlesPage;
